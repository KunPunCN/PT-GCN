import torch
from transformers.models.t5.modeling_t5 import T5LayerNorm
import torch.nn.functional as F
from .myutils import BertLinear, BertLayerNorm, gelu

class GAT(torch.nn.Module):
    def __init__(self, in_channel=768, out_channel=256, label_num=6):
        super(GAT2, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.label_num = label_num
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        dropout = 0.5
        # self.activation = torch.nn.GELU()
        self.dense1 = torch.nn.Linear(self.in_ch, self.in_ch)
        self.dense2 = torch.nn.Linear(self.in_ch, self.out_ch)

        layer_norm_eps = 1e-12
        self.norm1 = T5LayerNorm(in_channel, layer_norm_eps)
        self.norm2 = T5LayerNorm(self.out_ch, layer_norm_eps)

    def forward(self, table, a_simi, t_simi):  # [B,L,L,dim],[B,L],[B,L]

        batch, seq, seq, dim = table.shape
        a_simi2 = a_simi.unsqueeze(2).expand(batch, seq, seq).unsqueeze(3)  # [B,L,L,1]
        t_simi2 = t_simi.unsqueeze(1).expand(batch, seq, seq).unsqueeze(3)  # [B,L,L,1]

        h0 = table  # h_0
        h1 = self.gether(h0, a_simi2, t_simi2)
        h1 = self.relu(self.norm1(self.dense1(h1+h0)))  # h_1

        h2 = self.gether(h1, a_simi2, t_simi2)
        h2 = self.relu(self.norm2(self.dense2(h2+h1)))

        return h2

    def gether(self, h, a_simi, t_simi):
        batch, seq, seq, dim = h.shape
        padding_y = torch.zeros([batch, seq, 1, dim]).cuda()  # [B,1,L,dim]
        h_y = h * t_simi
        h_left = torch.cat([padding_y, h_y], dim=2)[:, :, :-1, :]  # [B,L+1,L,dim]
        h_right = torch.cat([h_y, padding_y], dim=2)[:, :, 1:, :]

        padding_x = torch.zeros([batch, 1, seq, dim]).cuda()  # [B,1,dim]
        h_x = h * a_simi
        h_up = torch.cat([padding_x, h_x], dim=1)[:, :-1, :, :]  # [B,L+1,L,dim]
        h_down = torch.cat([h_x, padding_x], dim=1)[:, 1:, :, :]

        h = h + h_left + h_right + h_up + h_down
        return h

class Simi(torch.nn.Module):
    def __init__(self, mlp_hidden_size=300):
        super(Simi, self).__init__()
        self.activation = torch.nn.GELU()
        mlp_size = mlp_hidden_size
        dropout = 0.5
        self.head_U3 = BertLinear(input_size=mlp_size * 3,
                                  output_size=mlp_size,
                                  activation=self.activation,
                                  dropout=dropout)
        self.tail_U3 = BertLinear(input_size=mlp_size * 3,
                                  output_size=mlp_size,
                                  activation=self.activation,
                                  dropout=dropout)

    def forward(self, h_a, h_t, aspect_simi, term_simi):  # [B,L,L,dim]
        h_a1 = self.atten_span3(h_a, aspect_simi, self.head_U3)
        h_t1 = self.atten_span3(h_t, term_simi, self.tail_U3)
        return h_a1, h_t1

    def atten_span3(self, h_a, aspect_simi, mlp):  # [B,L,d],[B,L]
        batch, lens, dim = h_a.shape
        # h_a0 = h_a
        h_a = h_a * aspect_simi.unsqueeze(2).expand_as(h_a)
        padding = torch.zeros([batch, 1, dim]).cuda()  # [B,1,dim]
        h_a_left = torch.cat([padding, h_a], dim=1)[:, :-1, :]  # [B,L+1,dim]
        h_a_right = torch.cat([h_a, padding], dim=1)[:, 1:, :]

        h_a1 = torch.cat([h_a_left, h_a, h_a_right], dim=-1)
        h_a1 = mlp(h_a1)
        return h_a1
