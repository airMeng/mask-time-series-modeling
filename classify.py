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
        src, src_mask,target = batch['data'].type(torch.float32).cuda(), batch['mask'].cuda(),batch['label'].type(torch.float32).cuda()
        out = model.forward(src, src_mask)
        print(out.shape)
        print(target.shape)
        loss=SimpleLossCompute(criterion,out,target,opt)
        opt.step()
        opt.optimizer.zero_grad()
        if i % 10 == 0:
            print(i, loss, opt._rate)


criterion=nn.MSELoss()
model = make_model()
model.cuda()
model_opt = get_std_opt(model)
data_loader=load_data()
for epoch in range(10):
    print('epoch : '+str(epoch))
    train_epoch(data_loader, model, criterion, model_opt)



model.eval()
data_test=np.load('./data/test/P-1.npy')
l_test=data_test.shape[0]


def save_data(name,data,x,y):
    np.savez(name,data,x,y)


for i in [1899,3289,4286]:
    d=data_test[i:(i+350),:]
    d_test=torch.FloatTensor(data_test[i:(i+200),:]).unsqueeze(0)
    mask=torch.from_numpy(np.ones((len(d_test),1))).type(torch.uint8)
    out=model.forward(d_test.cuda(),mask.cuda())
    for layer in range(0, 6):
    #fig, axs = plt.subplots(1,4, figsize=(20, 10))
        print("Encoder Layer", layer+1)
        #for h in range(4):
        name='./results/'+'data '+str(i)+' layer '+str(layer)+'.npz'
        h=0
        save_data(name,model.encoder.layers[layer].self_attn.attn[0, h].data.cpu(), 
            d, d if h ==0 else [])
    # plt.show()
def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)

