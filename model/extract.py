import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import len_mask
from .attention import prob_normalize

from transformers import BertModel

INI = 1e-2

# class LSTMEncoder(nn.Module):
#     def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
#         super().__init__()
#         self._init_h = nn.Parameter(torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
#         self._init_c = nn.Parameter(torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
#         init.uniform_(self._init_h, -INI, INI)
#         init.uniform_(self._init_c, -INI, INI)
#         self._lstm = nn.LSTM(input_dim, n_hidden, n_layer, dropout=dropout, bidirectional=bidirectional)
#
#     def forward(self, input_, in_lens=None):
#         """ [batch_size, max_num_sent, input_dim] Tensor"""
#         size = (self._init_h.size(0), input_.size(0), self._init_h.size(1)) # [n_layer, bs, n_hidden]
#         init_states = (self._init_h.unsqueeze(1).expand(*size),
#                        self._init_c.unsqueeze(1).expand(*size))
#         if in_lens is not None and not isinstance(in_lens, list):
#             in_lens = in_lens.tolist()
#         lstm_out, _ = lstm_encoder(input_, self._lstm, in_lens, init_states) # [bs, ns, n_hidden*2]
#         # print('the size of lstm_out is {}'.format(lstm_out.size()))
#         # print('the value of in_lens is {}'.format(in_lens))
#         return lstm_out.transpose(0, 1) # [seq_num, batch_size, emb_dim]
#
#     @property
#     def input_size(self):
#         return self._lstm.input_size
#
#     @property
#     def hidden_size(self):
#         return self._lstm.hidden_size
#
#     @property
#     def num_layers(self):
#         return self._lstm.num_layers
#
#     @property
#     def bidirectional(self):
#         return self._lstm.bidirectional


class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer, bidirectional=False)
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

    def forward(self, knowledge_state, knowledge_num, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""

        attn_feat, lstm_states, init_i = self._prepare(knowledge_state)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1) # (nt, bs, d)
        query, _ = self._lstm(lstm_in, lstm_states) # (nt, bs, d)
        query = query.transpose(0, 1) # [bs, nt, d]
        output = LSTMPointerNet.attention_score(attn_feat, query, self._attn_v, self._attn_wq)
        return output  # unormalized extraction logit

    def extract(self, knowledge_state, knowledge_num, k):
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, lstm_states, lstm_in = self._prepare(knowledge_state)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(self._lstm).to(knowledge_state.device)
        extracts = []
        for _ in range(k):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1].unsqueeze(1) # [bs, 1, d]
            score = LSTMPointerNet.attention_score(attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze() # vector, variable length
            for e in extracts:
                score[e] = -1e6
            ext = score.max(dim=0)[1].item()
            extracts.append(ext)
            lstm_states = (h, c)
            lstm_in = knowledge_state[:, ext, :]
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D])
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output


class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, lstm_hidden, lstm_layer, bert_config):
        super().__init__()
        self.transformer = BertModel.from_pretrained(bert_config)
        # self.transformer = BertModel.from_pretrained('/home/zhaoxl/Data/pretrain-models/bert-base-uncased')
        # self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self._extractor = LSTMPointerNet(emb_dim, lstm_hidden, lstm_layer)

    def forward(self, knowledge_input, knowledge_num, target):
        knowledge_state = torch.stack(
            [self._encode(knowledge_input[i]) for i in range(knowledge_input.size(0))], dim=0
        )
        bs, nt = target.size()
        d = knowledge_state.size(2)
        ptr_in = torch.gather(knowledge_state, dim=1, index=target.unsqueeze(2).expand(bs, nt, d))
        logit = self._extractor(knowledge_state, knowledge_num, ptr_in)
        return (logit, )

    def extract(self, knowledge_state, knowledge_num, k=1):
        output = self._extractor.extract(knowledge_state, knowledge_num, k)
        return output

    def _encode(self, input_ids):
        return self.transformer(input_ids)[0][:, 0]
