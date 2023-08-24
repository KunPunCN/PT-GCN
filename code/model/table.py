import torch
from torch import nn
from .seq2mat import *
from .tgcn.GAT import GAT


class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        if config.seq2mat == 'tensor':
            self.seq2mat = TensorSeq2Mat(config)
        elif config.seq2mat == 'tensorcontext':
            self.seq2mat = TensorcontextSeq2Mat(config)
        elif config.seq2mat == 'context':
            self.seq2mat = ContextSeq2Mat(config)
        else:
            self.seq2mat = Seq2Mat(config)

        self.gat = GAT()

    def forward(self, seq, mask, a_s, o_s):
        table = self.seq2mat(seq, seq)
        table = self.gat(table, a_s, o_s)

        return table
