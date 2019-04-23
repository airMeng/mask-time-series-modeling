import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src,src_mask,tgt,tgt_mask):
        return self.decoder(self.encoder(src,src_mask),src_mask,tgt,tgt_mask)

    def encoder(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)

    def decoder(self,memory,tgt,src_mask,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)


class Generator(nn.Module):
    def __init__(self,input_size,out_put_size):
        super(Generator, self).__init__()
        self.proj=nn.Linear(input_size,out_put_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x),-1)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self,layer,n):
        super(Encoder, self).__init__()
        self.layers=clones(layer,n)
        self.layernorm=LayerNorm(layer.size)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return self.layernorm(x)


class Decoder(nn.Module):
    def __init__(self,layer,n):
        super(Decoder, self).__init__()
        self.layers=clones(layer,n)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm, self).__init__()
        self.w=nn.Parameter(torch.ones(features))
        self.b=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self, x):
        x_mean=x.mean(-1,keepdim=True)
        x_std=x.std(-1,keepdim=True)
        return self.w*(x-x_mean)/(x_std+self.eps)+self.b


class SubLayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SubLayerConnection, self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=dropout

    def forward(self, x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self,size,feed_forward,self_attn,dropout):
        super(EncoderLayer, self).__init__()
        self.size=size
        self.feed_forward=feed_forward
        self.self_attn=self_attn
        self.dropout=dropout
        self.sublayer=clones(SubLayerConnection(size,dropout),2)

    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer, self).__init__()
        self.size=size
        self.feed_forward=feed_forward
        self.sublayer=clones(SubLayerConnection(size,dropout),3)
        self.self_attn=self_attn
        self.src_attn=src_attn

    def forward(self,x,memory,src_mask,tgt_mask):
        m=memory
        x=self.sublayer[0](x,lambda x :self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward(x))


def attention(query,key,value,mask=None,dropout=None):
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores=scores.mask_fill(mask==0,1e-9)
    p_attn=F.softmax(scores)
    if dropout is not None:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn


class MultiAttention(nn.Module):
    def __init__(self,h,d_model,dropout):
        "Take in model size and number of heads."
        super(MultiAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), h)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(2*max_len) / d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        self.dropout=dropout
        self.pe=pe.unsqueeze(0)

    def forward(self, x):
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=True)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiAttention(h, d_model,dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionEncoding(d_model,dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    print(model.parameters())
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for name,param in model.named_parameters():
        if param.requires_grad :
            print(name)
            print(param.dim())
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)
    return model

tmp_model = make_model(10, 10, 2)
op=torch.optim.Adam(tmp_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
print(op.param_groups)


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0.
    total_loss = 0
    tokens = 0.
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens.float()
        tokens += batch.ntokens.float()
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens.float(), tokens / elapsed))
            start = time.time()
            tokens = 0.
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

a=get_std_opt(tmp_model)
a.optimizer.zero_grad()
