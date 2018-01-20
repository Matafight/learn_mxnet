#_*_coding:utf-8_*_

'''
1. need validation and testing data
2. need to support checkpoint
3. draw the curve of  validation and training error 
4. consider using stop and begin words
5. how to better proprecessing the input data?
'''
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn, autograd
from poem_util import transform_data, generate_batch, train_test_split
import time
import numpy as np
class RNNModel(gluon.Block):
    def __init__(self,mode,vocab_size,embed_dim,hidden_dim,num_layers,dropout=0.5,**kwargs):
        super(RNNModel,self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            #self.encoder = nn.Embedding(input_dim=vocab_size,output_dim = embed_dim,weight_initializer=mx.init.Uniform(0,1))
            self.encoder = nn.Embedding(input_dim=vocab_size,output_dim = embed_dim)

            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(hidden_dim,num_layers,activation='relu',dropout=dropout,input_size = embed_dim)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(hidden_dim,num_layers,dropout=dropout,input_size=embed_dim)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(hidden_dim,num_layers,dropout = dropout,input_size=embed_dim)
            elif mode =='gru':
                self.rnn = rnn.GRU(hidden_dim,num_layers,dropout=dropout,input_size = embed_dim)
            else:
                raise ValueError("Invalid mode %s.optiions are rnn_relu,rnn_tanh,lstm and gru"%mode)
            self.decoder = nn.Dense(vocab_size,in_units=hidden_dim)
            self.hidden_dim = hidden_dim

    def forward(self,inputs,state):
        emb = self.drop(self.encoder(inputs))
        output,state = self.rnn(emb,state)
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
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size,
                                   ctx = context)
        #get training  iterator here
        for ibatch,(data,target) in enumerate(train_iter):
            data = data.T
            #I need to make sure it is stacked in the correct orientation, yes,it's right
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
            if ibatch % eval_period == 0 and ibatch > 0:
                cur_L = total_L / num_steps / batch_size / eval_period
                print('[Epoch %d ] loss %.2f, perplexity %.2f' % (
                    epoch + 1, cur_L, np.exp(cur_L)))
                total_L = 0.0

        val_L = model_eval(test_iter)
        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation '
              'perplexity %.2f' % (epoch + 1, time.time() - start_time, val_L,
                                   np.exp(val_L)))

# 定义参数
model_name = 'gru'
embed_dim = 100
hidden_dim = 100
num_layers = 1
lr = 0.1
clipping_norm = 0.2
epochs=1
batch_size = 64
num_steps = 90
dropout_rate = 0.2
eval_period = 1

context = try_gpu()
corpus_vec,word_to_int = transform_data('../input/poems.txt')
vocab_size = len(word_to_int)
training_vec,testing_vec= train_test_split(corpus_vec)

model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim,
                       num_layers, dropout_rate)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

testing_iter = generate_batch(testing_vec,word_to_int,batch_size,ctx=context)
training_iter = generate_batch(training_vec,word_to_int,batch_size,ctx=context)
print('start training...\n')
train_and_eval(training_iter,testing_iter)