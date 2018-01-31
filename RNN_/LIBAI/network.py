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
import mxnet.ndarray as nd
from mxnet.gluon import nn, rnn, autograd
from poem_util import transform_data, generate_batch, train_test_split,ReusableGenerator
import time
import numpy as np
class RNNModel(gluon.Block):
    def __init__(self,mode,vocab_size,embed_dim,hidden_dim,num_layers,dropout=0.5,**kwargs):
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
        start = time.time()
        #training iter needs reset
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
            if ibatch % eval_period == 0 and ibatch > 0:
                cur_L = total_L / num_steps / batch_size / eval_period
                print('[Epoch %d ] loss %.2f, perplexity %.2f' % (
                    epoch + 1, cur_L, np.exp(cur_L)))
                total_L = 0.0
        end = time.time()
        #print('Epoch:%d, elapsed:%s'%(epoch,end-start))
        #val_L = model_eval(test_iter)
        inference_from_word(model,"春",10,hidden_dim,embed_dim,word_to_int,context)
        if epoch%save_period ==0:
            filenames = checkpoint_path+'epoch_'+str(epoch)+'.params'
            model.save_params(filenames)
        #print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation '
        #      'perplexity %.2f' % (epoch + 1, time.time() - start_time, val_L,
        #                           np.exp(val_L)))

# update on 1/26/2018, currently, we only support one word,we will support multi-words in the near future
def inference_from_word(rnn,prefix,num_word,hidden_dim,embed_dim,word_to_int,ctx=mx.cpu()):
    vocab_size = len(word_to_int)
    hidden = nd.zeros(shape = (1,1,hidden_dim),ctx=ctx)
    #假设输入的样本是batch_size = 1,num_steps = 1
    prefix = word_to_int[prefix]
    print('prefix:%f'%prefix)
    prefix = nd.array([[prefix]],ctx=ctx)
    outputs = []
    for i in range(num_word):
        output,hidden = rnn(prefix,hidden)
        #get words based on the current output with shape(1,vocab_size)
        output_idx = int(nd.argmax(output,axis=1).asscalar())+1
        prefix = nd.array([[output_idx]],ctx=ctx)
        outputs.append(output_idx)
    #get words from index
    words = [int_to_word[i] for i in outputs]       
    print(words)
    
    



# 定义参数
model_name = 'gru'
embed_dim = 100
hidden_dim = 50
num_layers = 1
lr = 0.1
clipping_norm = 0.2
epochs=10
batch_size = 64
num_steps = 90
dropout_rate = 0.2
eval_period = 15
checkpoint_path = './checkpoints/'
save_period = 5

context = try_gpu()
corpus_vec,word_to_int,int_to_word = transform_data('../input/poems.txt',num_steps)
vocab_size = len(word_to_int)
training_vec,testing_vec= train_test_split(corpus_vec)

model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim,
                       num_layers, dropout_rate)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

testing_iter =  ReusableGenerator(generate_batch,testing_vec,word_to_int,batch_size,ctx=context)
training_iter = ReusableGenerator(generate_batch,training_vec,word_to_int,batch_size,ctx=context)
print('start training...\n')
train_and_eval(training_iter,testing_iter)