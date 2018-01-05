# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 19:26:54 2018

@author: guo
"""
from mxnet.gluon import nn,Block
from mxnet import ndarray as F
import mxnet as mx

class model(Block):
    def __init__(self,**kwargs):
        super(model,self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(20)
            self.dense1 = nn.Dense(20)
            #dense1与dense2 共享参数
            #self.dense1 = nn.Dense(20,params = self.dense0.collect_params())
    def forward(self,X):
        x = F.relu(self.dense0(X))
        return F.relu(self.dense1(x))
#自定义前缀
model =  model(prefix='net_')
model.initialize(ctx=mx.cpu(0))
model(F.zeros((10,10),ctx=mx.cpu(0)))
all_params = model.collect_params()
print(all_params)

#输出参数的值
print(all_params['net_dense0_bias'].data())
        