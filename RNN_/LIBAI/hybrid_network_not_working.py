#_*_coding:utf-8_*_
'''
to support checkpont,we need to write hybrid version of network
not working because gluon not support hybridRNN now
'''
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn, autograd
import mxnet.ndarray as nd
from poem_util import transform_data, generate_batch, train_test_split
import time
import numpy as np
class RNNModel(rnn.HybridRecurrentCell):
    def __init__(self,mode,vocab_size,embed_dim,hidden_dim,num_layers,num_steps,dropout=0.5,**kwargs):
        super(RNNModel,self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(input_dim=vocab_size,output_dim = embed_dim)
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(hidden_dim,num_layers,activation='relu',dropout=dropout,input_size = embed_dim)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(hidden_dim,num_layers,dropout=dropout,input_size=embed_dim)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(hidden_dim,num_layers,dropout = dropout,input_size=embed_dim)
            elif mode =='gru':
                self.rnn = rnn.GRUCell(hidden_size=hidden_dim,input_size=embed_dim)
                #self.rnn = rnn.GRU(hidden_dim,num_layers,dropout=dropout,input_size = embed_dim)
            else:
                raise ValueError("Invalid mode %s.optiions are rnn_relu,rnn_tanh,lstm and gru"%mode)
            self.decoder = nn.Dense(vocab_size,in_units=hidden_dim)
            self.hidden_dim = hidden_dim
            self.num_steps = int(num_steps)

    def hybrid_forward(self,F,inputs,state):
        #F is symbol when hybridized, otherwise ndarray
        emb = self.drop(self.encoder(inputs))
        # the question is that the sym may not has the shape method,how to handle with this situatation??
        # solution 1:fixed the num_steps
        output = []
        #opps! for loop is not supported in hybridnetwork
        for i in range(self.num_steps):
            cur_output,state = self.rnn(emb[i],state)
            output.append(cur_output)
        output = nd.array(output)
        #output,state = self.rnn(emb,state)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1,self.hidden_dim)))
        return (decoded,state)

    def begin_state(self,*args,**kwargs):
        return self.rnn.begin_state(*args,**kwargs)

def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state

def try_gpu():
    try:
        ctx=mx.gpu()
        _ = nd.array([0],ctx=ctx)
    except:
        ctx=mx.cpu()
    return ctx

def model_eval(data_iter):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size,
                               ctx=context)
    for data,label in data_iter:
        data = data.T
        label = label.T.reshape((-1,))
        output,hidden = model(data,hidden)
        L = loss(output,label)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L/ntotal

def train_and_eval(train_iter,test_iter):
    for epoch in range(epochs):
        total_L = 0.0
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size,
                                   ctx = context)
        for ibatch,(data,target) in enumerate(train_iter):
            print('i\n')
            #(batch_size,num_steps)->(num_steps,batch_size)
            data = data.T
            #num_steps个batch_size
            target = target.T.reshape((-1,))
            # 从计算图分离隐含状态。
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。
            # 因此我们将clipping_norm乘以num_steps和batch_size。
            gluon.utils.clip_global_norm(grads,
                                         clipping_norm * num_steps * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()
            print(total_L)

def inference():
    pass

# 定义参数
model_name = 'gru'
embed_dim = 100
hidden_dim = 100
num_layers = 1
lr = 0.1
clipping_norm = 0.2
epochs=1
batch_size = 64
num_steps = 50
dropout_rate = 0.2
eval_period = 1

context = try_gpu()
corpus_vec,word_to_int,int_to_word = transform_data('../input/poems.txt',num_steps)
vocab_size = len(word_to_int)
training_vec,testing_vec= train_test_split(corpus_vec)
num_steps =  max(map(len,corpus_vec)) 

model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim,
                       num_layers, num_steps,dropout_rate)
model.hybridize()
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
testing_iter = generate_batch(testing_vec,word_to_int,batch_size,ctx=context)
training_iter = generate_batch(training_vec,word_to_int,batch_size,ctx=context)
print('start training...\n')
train_and_eval(training_iter,testing_iter)