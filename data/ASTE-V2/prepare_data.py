import os
import torch
from transformers import BertTokenizer
import json


def syn_matix(tokens, tgt_text, trainedTokenizer, label_pair, senti):
    token_range = []
    mask_position = []
    token_start = 0
    for i, w, in enumerate(tokens):

        token_end = token_start + len(trainedTokenizer.encode(w, add_special_tokens=False))
        token_range.append([token_start, token_end])
        if w == '[MASK]':
            mask_position.append(token_start)
        token_start = token_end

    aspect = tgt_text[0].split('; ')[1:]  #
    term = tgt_text[1].split('; ')[1:]  #

    if aspect != [] and term != []:
        for i, j in zip(aspect, term):
            i = i.split(', ')
            j = j.split(', ')
            a_start = token_range[int(i[0])][0]
            a_end = token_range[int(i[-1])][1]
            o_start = token_range[int(j[0])][0]
            o_end = token_range[int(j[-1])][1]
            label_pair.append([a_start, a_end, o_start, o_end, senti])

    return label_pair, mask_position


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for file_name in ['14lap', '14res', '15res', '16res']:
    dataset = {}
    for split in ['train', 'test', 'dev']:
        dataset[split] = []
        filename = split + '_triplets.txt'
        jsonname = split + '.json'
        data_path = os.path.join(file_name, filename)
        json_path = os.path.join(file_name, jsonname)

        with open(data_path, 'r', encoding='utf-8') as f:
            with open(json_path, 'a', encoding='utf8') as f2:
                line = f.readline()
                idx = 0
                dict_list = []
                while line:
                    aspect = {}
                    term = {}
                    text_a = line.split('####')[0]
                    text_a = text_a + ' positive [MASK] [MASK] , neutral [MASK] [MASK] , negative [MASK] [MASK]'#11
                    tgt_text = line.split('####')[1].strip('####[(').strip(')]\n').split('), (')

                    for x in ['POS', 'NEG', 'NEU']:
                        aspect[x] = ''
                        term[x] = ''
                    for i in tgt_text:
                        (a, b, c) = i.split('], ')
                        a = a.strip('[')
                        b = b.strip('[')
                        if c == '\'POS\'':
                            label = 'POS'
                        elif c == '\'NEG\'':
                            label = 'NEG'
                        elif c == '\'NEU\'':
                            label = 'NEU'
                        aspect[label] = aspect[label] + '; ' + a
                        term[label] = term[label] + '; ' + b

                    label_pair = []
                    for i, l in enumerate(['POS', 'NEG', 'NEU']):
                        input_example = {}
                        tgt_text = [aspect[l], term[l]]
                        if aspect[l].split('; ')[1:] != [] and term[l].split('; ')[1:] != []:
                            label_pair, mask_position = syn_matix(text_a.split(' '), tgt_text, tokenizer, label_pair, l)

                    input_example = {}
                    input_example["ID"] = idx
                    input_example["sentence"] = text_a
                    input_example["pairs"] = label_pair
                    input_example["mask_position"] = mask_position
                    input_example["tokens"] = tokenizer.convert_ids_to_tokens(
                        tokenizer(text_a, add_special_tokens=False)['input_ids'])

                    dict_list.append(input_example)
                    idx += 1
                    line = f.readline()
                json.dump(dict_list, f2)
                f2.close()
            f.close()
