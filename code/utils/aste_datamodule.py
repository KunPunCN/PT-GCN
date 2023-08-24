import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
import os
from . import load_json

import random
from torch.utils.data.sampler import Sampler

polarity_map = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}

polarity_map_reversed = {
    0: 'NEG',
    1: 'NEU',
    2: 'POS'
}


class Example:
    def __init__(self, data, max_length=-1):
        self.data = data
        self.max_length = max_length
        self.data['tokens'] = eval(str(self.data['tokens']))

    def __getitem__(self, key):
        return self.data[key]

    def table_label(self, length, ty, id_len):
        label = [[-1 for _ in range(length)] for _ in range(length)]
        id_len = id_len.item()-9

        for i in range(1, id_len - 1):
            for j in range(1, id_len - 1):
                label[i][j] = 0
        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if ty == 'S':
                label[t_start + 1][o_start + 1] = 1
            elif ty == 'E':
                label[t_end][o_end] = 1
        return label

    def seq_label(self, length, ty, id_len):
        label = [-1 for _ in range(length)]
        id_len = id_len.item()-7

        for i in range(1, id_len - 1):
            label[i] = 0
        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if ty == 'S':
                for j in range(t_start+1, t_end+1):
                    label[j] = 2
                label[t_start + 1] = 1
            elif ty == 'E':
                for j in range(o_start+1, o_end+1):
                    label[j] = 2
                label[o_start + 1] = 1
        return label

    def mask_label(self, length, id_len):
        label = [0 for _ in range(length)]
        # id_len = id_len.item()
        for i in self['mask_position']:
                label[i+1] = 1
        return label

    def set_pairs(self, pairs):
        self.data['pairs'] = pairs



class DataCollatorForASTE:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples):

        batch = self.tokenizer_function(examples)

        length = batch['input_ids'].size(1)

        batch['example_ids'] = [example['ID'] for example in examples]
        batch['table_labels_S'] = torch.tensor(
            [examples[i].table_label(length, 'S', (batch['input_ids'][i] > 0).sum()) for i in range(len(examples))],
            dtype=torch.long)
        batch['table_labels_E'] = torch.tensor(
            [examples[i].table_label(length, 'E', (batch['input_ids'][i] > 0).sum()) for i in range(len(examples))],
            dtype=torch.long)
        batch['mask_position'] = torch.tensor(
            [examples[i].mask_label(length, (batch['input_ids'][i] > 0).sum()) for i in range(len(examples))],
            dtype=torch.long)

        al = [example['pairs'] for example in examples]
        pairs_ret = []
        for pairs in al:
            pairs_chg = []
            for p in pairs:
                pairs_chg.append([p[0], p[1], p[2], p[3], polarity_map[p[4]] + 1])
            pairs_ret.append(pairs_chg)
        batch['pairs_true'] = pairs_ret

        return {
            'ids': batch['example_ids'],
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],

            'table_labels_S': batch['table_labels_S'],
            'table_labels_E': batch['table_labels_E'],
            'mask_position': batch['mask_position'],

            'pairs_true': batch['pairs_true'],
        }

    def tokenizer_function(self, examples):
        text = [example['sentence'] for example in examples]
        kwargs = {
            'text': text,
            'return_tensors': 'pt'
        }

        if self.max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True
        else:
            kwargs['padding'] = 'max_length'
            kwargs['max_length'] = self.max_seq_length
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)

        batch_encodings = dict(batch_encodings)

        return batch_encodings


class ASTEDataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path: str = '',
                 max_seq_length: int = -1,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 num_workers: int = 4,
                 cuda_ids: int = -1,
                 ):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length if max_seq_length > 0 else 'longest'
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.cuda_ids = cuda_ids

        self.table_num_labels = 6  # 4

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name = os.path.join(self.data_dir, 'dev.json')
        test_file_name = os.path.join(self.data_dir, 'test.json')

        if not os.path.exists(dev_file_name):
            dev_file_name = test_file_name

        train_examples = [Example(data, self.max_seq_length) for data in load_json(train_file_name)]
        dev_examples = [Example(data, self.max_seq_length) for data in load_json(dev_file_name)]
        test_examples = [Example(data, self.max_seq_length) for data in load_json(test_file_name)]

        self.raw_datasets = {
            'train': train_examples,
            'dev': dev_examples,
            'test': test_examples
        }

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForASTE(tokenizer=self.tokenizer,
                                           max_seq_length=self.max_seq_length),
            pin_memory=True,
            prefetch_factor=16
        )
        return dataloader


    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)