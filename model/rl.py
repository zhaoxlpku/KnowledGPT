import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .extract import LSTMPointerNet


INI = 1e-2
# INI = 1e-1

class PtrExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        self._stop = nn.Parameter(torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._stop, -INI, INI) # check INI


    def forward(self, knowledge_state, knowledge_lens, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        # max_step = knowledge_state.size(0)
        # knowledge_state = torch.cat([knowledge_state, self._stop.unsqueeze(0)], dim=0)

        attn_feat = torch.matmul(knowledge_state, self._attn_wm.unsqueeze(0)) # [batch, ns, hidden]
        outputs = []
        dists = []
        bs, ns, dim = knowledge_state.size()
        n_layer, hidden = self._init_h.size()
        size = (n_layer, bs, hidden)
        lstm_in = self._init_i.unsqueeze(0).expand(bs, dim) # [batch, dim]
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous()) # [layer, 1, hidden]
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1].unsqueeze(1) # [batch, 1, hidden]
            score = PtrExtractorRL.attention_score(attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze(1) # [batch, ns]
            assert score.size() == (bs, ns)
            mask = (torch.arange(ns).expand(bs, ns) >= torch.tensor(knowledge_lens).unsqueeze(1))
            mask = mask.type_as(score)
            score = score + (mask * -1e20)

            if self.training:
                prob = F.softmax(score, dim=-1) # [bs, ns]
                m = torch.distributions.Categorical(prob)
                dists.append(m) # tensor, [bs, ns]
                out = m.sample() # tensor, [bs]
            else:
                out = score.max(dim=-1)[1] # tensor, [bs]
            outputs.append(out)
            # if out.item() == max_step:
            #     break

            lstm_in = torch.gather(knowledge_state, dim=1, index=out.unsqueeze(1).unsqueeze(2).expand(bs, 1, dim))
            lstm_in = lstm_in.squeeze(1) # [batch, dim]
            lstm_states = (h, c)
        if dists:
            # return distributions only when not empty (training)
            return outputs, dists
        else:
            return outputs

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


class PolicyGradient(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, transformer, extractor):
        super().__init__()
        self._transformer = transformer
        self._extract = PtrExtractorRL(extractor)
        # self._batcher = art_batcher

    def _encode(self, input_ids):
        return self._transformer(input_ids)[0][:, 0]

    def forward(self, knowledge_input, knowledge_lens, n_sent):
        bs, ns, trunc = knowledge_input.size()
        knowledge_input = knowledge_input.view(bs * ns, trunc)
        knowledge_state = self._encode(knowledge_input)
        knowledge_state = knowledge_state.view(bs, ns, 768)

        outputs = self._extract(knowledge_state, knowledge_lens, n_sent)
        return outputs
