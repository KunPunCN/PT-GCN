import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from .table import TableEncoder, Ptgcn
from .matching_layer import MatchingLayer
from torch.nn import functional as F
from torch.autograd import Function
from typing import Any, Optional, Tuple
class BDTFModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.table_encoder = TableEncoder(config)
        self.gcn1 = Ptgcn(config)
        self.gcn2 = Ptgcn(config)
        self.gcn3 = Ptgcn(config)
        self.inference = InferenceLayer(config)
        self.matching = MatchingLayer(config)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(2, 768, 768))
        self.w1.data.zero_()
        self.t1 = torch.nn.Parameter(torch.FloatTensor(2, 768, 768))
        self.t1.data.zero_()
        self.w2 = torch.nn.Parameter(torch.FloatTensor(2, 768, 768))
        self.w2.data.zero_()
        self.t2 = torch.nn.Parameter(torch.FloatTensor(2, 768, 768))
        self.t2.data.zero_()
        self.w3 = torch.nn.Parameter(torch.FloatTensor(2, 768, 768))
        self.w3.data.zero_()
        self.t3 = torch.nn.Parameter(torch.FloatTensor(2, 768, 768))
        self.t3.data.zero_()

        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.init_weights()

    def forward(self, input_ids, attention_mask, ids,
                mask_position=None,table_labels_S=None, table_labels_E=None,
                polarity_labels=None, pairs_true=None, domain_id=None):
        seq = self.bert(input_ids, attention_mask)[0]

        table_mask = torch.where(table_labels_S>=0, 1, 0)
        attention_mask = table_mask[:,1,:]
        mask_position = torch.where(mask_position >= 1, True, False)
        batch, l, dim = seq.shape
        outputs_at_mask = torch.masked_select(seq, mask_position.unsqueeze(-1))
        outputs_at_mask = outputs_at_mask.view(batch, -1, dim)
        as1, as2, as3, ts1, ts2, ts3 = self.get_atten(outputs_at_mask, seq, attention_mask)
        seq = seq * (attention_mask.unsqueeze(-1))

        table = self.table_encoder(seq)
        table1 = self.gcn1(table, as1, ts1)
        table2 = self.gcn2(table, as2, ts2)
        table3 = self.gcn3(table, as3, ts3)
        table1 = torch.cat([table1,table2,table3], dim=-1)

        output = self.inference(table1, attention_mask, table_labels_S, table_labels_E)
        output['ids'] = ids

        output = self.matching(output, table1, pairs_true, seq)

        return output
    
    def get_atten(self,outputs_at_mask,seq,attention_mask):
        am1, tm1, am2, tm2, am3, tm3 = torch.chunk(outputs_at_mask, 6, dim=1)
        as1 = self.attention_dot(seq, am1, attention_mask, self.w1).to(torch.float)
        ts1 = self.attention_dot(seq, tm1, attention_mask, self.t1).to(torch.float)
        as2 = self.attention_dot(seq, am2, attention_mask, self.w2).to(torch.float)
        ts2 = self.attention_dot(seq, tm2, attention_mask, self.t2).to(torch.float)
        as3 = self.attention_dot(seq, am3, attention_mask, self.w3).to(torch.float)
        ts3 = self.attention_dot(seq, tm3, attention_mask, self.t3).to(torch.float)
        return as1, as2, as3, ts1, ts2, ts3


    def attention_dot(self, embed, prob_attention, attention_mask, w1):
        simi = torch.einsum('bxi, oij, byj -> boxy', embed, w1, prob_attention).squeeze(-1).permute(0, 2, 1)
        simi = torch.softmax(simi, dim=-1)
        simi = simi[:, :, 0].squeeze(-1) * attention_mask
        return simi  # [batch_size,seq_length]



class InferenceLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_linear_S = nn.Linear(768, 1)
        self.cls_linear_E = nn.Linear(768, 1)

    def span_pruning(self, pred, z, attention_mask):
        mask_length = attention_mask.sum(dim=1) - 2
        length = ((attention_mask.sum(dim=1) - 2) * z).long()
        length[length < 5] = 5
        max_length = mask_length ** 2
        for i in range(length.shape[0]):
            if length[i] > max_length[i]:
                length[i] = max_length[i]
        batch_size = attention_mask.shape[0]
        pred_sort, _ = pred.view(batch_size, -1).sort(descending=True)  # 降序
        batchs = torch.arange(batch_size).to('cuda')
        topkth = pred_sort[batchs, length - 1].unsqueeze(1)
        return pred >= (topkth.view(batch_size, 1, 1))

    def forward(self, table, attention_mask, table_labels_S, table_labels_E):
        outputs = {}

        logits_S = torch.squeeze(self.cls_linear_S(table), 3)
        logits_E = torch.squeeze(self.cls_linear_E(table), 3)

        mask = table_labels_S
        loss_func = nn.BCEWithLogitsLoss(weight=(mask >= 0))
        losss = loss_func(logits_S, table_labels_S.float())
        losse = loss_func(logits_E, table_labels_E.float())
        outputs['table_loss'] = losse+losss

        S_pred = torch.sigmoid(logits_S) * (table_labels_S >= 0)
        E_pred = torch.sigmoid(logits_E) * (table_labels_S >= 0)

        table_predict_S = self.span_pruning(S_pred, self.config.span_pruning, attention_mask)
        table_predict_E = self.span_pruning(E_pred, self.config.span_pruning, attention_mask)


        outputs['S_prob'] = S_pred
        outputs['E_prob'] = E_pred
        outputs['logits_S'] = logits_S
        outputs['logits_E'] = logits_E
        outputs['table_predict_S'] = table_predict_S
        outputs['table_predict_E'] = table_predict_E
        outputs['table_labels_S'] = table_labels_S
        outputs['table_labels_E'] = table_labels_E
        return outputs
