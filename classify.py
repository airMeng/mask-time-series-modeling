import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from encoder import make_model
from read_data import *

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


def SimpleLossCompute(criterion,x,y,opt):

    loss = criterion(x.contiguous().view(-1, x.size(-1)),
                          y.contiguous().view(-1))
    loss.backward()
    if opt is not None:
        opt.step()
        opt.optimizer.zero_grad()
    return loss.item()


def train_epoch(train_iter, model, criterion, opt, transpose=False):
    model.train()
    for i, batch in enumerate(train_iter):
        src, src_mask,target = batch['data'].type(torch.float32), batch['mask'],batch['label']
        out = model.forward(src, src_mask)
        loss=SimpleLossCompute(criterion,out,target,opt)
        opt.step()
        opt.optimizer.zero_grad()
        if i % 10 == 1:
            print(i, loss, opt._rate)


criterion=nn.MSELoss()
model = make_model()
model_opt = get_std_opt(model)
data_loader=load_data()
for epoch in range(20):
    print('epoch : '+str(epoch))
    train_epoch(data_loader, model, criterion, model_opt)