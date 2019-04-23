import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from encoder import make_model

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
    return NoamOpt(model.generator.d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def loss_backprop(criterion, out, targets):
    """
    Memory optmization. Compute each timestep separately and sum grads.
    """
    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []
    out_column=Variable(out.data,requires_grad=True)
    loss = criterion(out_column, targets)
    total += loss.item()
    loss.backward()
    out_grad.append(out_column.grad.data.clone())
    out_grad = torch.stack(out_grad, dim=1)
    out.backward(gradient=out_grad)
    return total


def SimpleLossCompute(criterion,x,y,opt):

    loss = criterion(x.contiguous().view(-1, x.size(-1)),
                          y.contiguous().view(-1))
    loss.backward()
    if opt is not None:
        opt.step()
        opt.optimizer.zero_grad()
    return loss.item()


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(src, pad):
    #src[:, 2:4] = 0
    #src_mask = (src.sum(0) != pad).unsqueeze(-2)
    # src_mask = src_mask & Variable(subsequent_mask(src.size(-1)).type_as(src_mask.data))
    src_mask=torch.tensor([1,1,0,0,1,1,1,1,1,1],dtype=torch.uint8)
    return src_mask


def train_epoch(train_iter, model, criterion, opt, transpose=False):
    model.train()
    for i, batch in enumerate(train_iter):
        src, src_mask = \
            batch.src, batch.src_mask
        src_masked=src.clone()
        src_masked[:,2:4,:]=0
        target=torch.mul(src[:,:,-1],(1-src_mask).type(torch.float32)).unsqueeze(-1)
        out = model.forward(src_masked, src_mask)
        #loss = loss_backprop(criterion, out, target)
        loss=SimpleLossCompute(criterion,out,target,opt)
        # opt.step()
        # opt.optimizer.zero_grad()
        if i % 10 == 1:
            print(i, loss, opt._rate)


class Batch:
    def __init__(self, src, src_mask):
        self.src = src
        self.src_mask = src_mask


def data_gen(batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randn(batch, 10,5)).type(torch.FloatTensor)
        src = Variable(data, requires_grad=False)
        #src[:,:,2:4]=0
        src_mask = make_std_mask(src, 0)
        print(src)
        print(src_mask)
        yield Batch(src, src_mask)

V=5
#criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
criterion=nn.MSELoss()
model = make_model()
model_opt = get_std_opt(model)
for epoch in range(20):
    print('epoch : '+str(epoch))
    train_epoch(data_gen(5, 5), model, criterion, model_opt)


model.eval()
src=torch.from_numpy(np.random.randn(1, 10,5)).type(torch.FloatTensor)
src_mask = make_std_mask(src, 0)
src_masked=src.clone()
src_masked[:,2:4,:]=0
target=torch.mul(src[:,:,-1],(1-src_mask).type(torch.float32)).unsqueeze(-1)
out = model.forward(src_masked, src_mask)
print('===========================================================================')
print(src)
print(out)