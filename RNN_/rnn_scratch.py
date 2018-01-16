#_*_coding:utf-8_*_

import random
import mxnet as mx
import numpy as np
import sys
sys.path.append('..')
import util
import mxnet.ndarray as nd


def load_data():
    with open('./input/jaychou_lyrics.txt','r',encoding='utf-8') as fh:
        lines = fh.read()
        print(len(lines))
    #将list中所有元素拼接成一个长度字符串
    content = lines.replace('\n',' ').replace('\r',' ')
    # construct data iterator 
    content = content[0:20000]
    idx_to_char = list(set(content))
    char_to_idx = dict([(char,int(i)) for i,char in enumerate(idx_to_char)])
    vocab_size = len(idx_to_char)
    idx_corpus = [char_to_idx[c] for c in content]
    return content,idx_to_char,char_to_idx,idx_corpus,vocab_size



def data_iter_random(num_steps,batch_size,corpus,ctx=mx.cpu()):
    #随机生成 data iterator，用yield返回？？
    #先随机分段
    #再随机选择batch_size个字段
    len_corpus=len(corpus)
    num_patch = len_corpus//num_steps
    num_batch = num_patch//batch_size 
    idx_list = [x for x in range(num_patch)]
    random.shuffle(idx_list)
    
    for batch in range(num_batch):
        #在idx_list中随机选择batch_size 个元素，选完之后在idx_list中删除这些元素
        to_use = random.sample(idx_list,batch_size)
        batch_data = []
        batch_label = []
        for item in to_use:
            idx_list.remove(item)
        for step in to_use:
            #i * num_patch ~ (i+1)*num_path
            data = corpus[step*num_steps:(step+1)*num_steps]
            batch_data.append(data)
            #判断最后一个是否越界
            if (step+1)*num_steps < len_corpus:
                label = corpus[step*num_steps + 1:(step+1)*num_steps+1]
            else:
                label = corpus[step*num_steps+1:(step+1)*num_steps]
                label.append('0')#填充字符
            batch_label.append(label)
        #聚合
        #ndarray batch_size * num_steps
        batch_data = mx.ndarray.array(batch_data)
        batch_label = mx.ndarray.array(batch_label)
        yield(batch_data,batch_label)



def data_iter_consective(num_steps,batch_size,corpus,ctx=mx.cpu()):
    #连续采样
    num_patch = len(corpus)//num_steps
    num_batch = num_patch//batch_size
    for i_batch in range(num_batch):
        total_data = []
        total_label = []
        for i_batch_size in range(batch_size):
            start_pos = i_batch+num_batch*(i_batch_size)
            data = corpus[start_pos*num_steps:(start_pos+1)*num_steps]
            #判断最后是否越界
            if(start_pos+1)*num_steps < len(corpus):
                label = corpus[start_pos*num_steps+1:(start_pos+1)*num_steps+1]
            else:
                label = corpus[start_pos*num_steps+1:(start_pos+1)*num_steps]
                label.append('0')
            total_data.append(data)
            total_label.append(label)
        total_data = mx.ndarray.array(total_data)
        total_label = mx.ndarray.array(total_label)
        yield(total_data,total_label)

def one_hot(data,vocab_size):
    inp = [nd.one_hot(x,vocab_size) for x in data.T]
    return inp

def get_params(input_dim,hidden_dim,output_dim,ctx=mx.cpu()):
    #set params
    std = .01
    W_xh= nd.random_normal(scale=std,shape=(input_dim,hidden_dim),ctx=ctx)
    W_hh = nd.random_normal(scale=std,shape=(hidden_dim,hidden_dim),ctx=ctx)
    b_h = nd.zeros(hidden_dim,ctx=ctx)
    
    W_hy = nd.random_normal(scale = std,shape=(hidden_dim,output_dim),ctx=ctx)
    b_y = nd.zeros(output_dim,ctx=ctx)
    params = [W_xh,W_hh,b_h,W_hy,b_y]
    for param in params:
        param.attach_grad()
    return params

def rnn(inputs,states,*params):
    #inputs  is a list shape = batch_size * input_dim
    H = states
    W_xh,W_hh,b_h,W_hy,b_y = params
    outputs = [] 
    for X in inputs:
        H = nd.tanh(nd.dot(X,W_xh)+nd.dot(H,W_hh)+b_h)
        Y = nd.tanh(nd.dot(H,W_hy)+b_y)
        outputs.append(Y)
    return (outputs,H)

#training and inference

def predict_rnn(rnn,prefix,vocab_size,num_char,params,hidden_dim,char_to_idx,idx_to_char):
    H = nd.zeros(shape=(1,hidden_dim),ctx=ctx)
    outputs = []
    print(char_to_idx[prefix[0]])
    outputs.append(char_to_idx[prefix[0]])
    
    for i in range(num_char+len(prefix)):
        x = nd.array([outputs[-1]],ctx=ctx)
        x = one_hot(x,vocab_size)
        out, H = rnn(x,H,*params)
        if i< len(prefix)-1:
            new_input = char_to_idx[prefix[i+1]]
        else:
            new_input  = int(out[-1].argmax(axis=1).asscalar())
        outputs.append(new_input)
    print(outputs)
    return ''.join([idx_to_char[x] for x in outputs])

# gradient cliping
def grad_clipping(params,theta,ctx= mx.cpu()):
    #python传参是传的引用，所有不用返回值
    if theta is not None:
        norm = nd.array([0.0],ctx=ctx)
        for p in params:
            norm = norm + nd.sum(p.grad**2)
        norm = nd.sqrt(norm).asscalar()
        if norm > theta:
            for p in params:
                p.grad[:] *= theta/norm

# training
#perplexity 困惑度
def train_and_predict(rnn_network,get_params,num_steps,idx_corpus,num_epochs=20,batch_size=64,vocab_size=256,hidden_dim = 128,learning_rate = 0.1,ctx=mx.cpu()):
    #定义损失
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    #生成数据迭代器
    #循环num_epoch
    
    params = get_params(vocab_size,hidden_dim,vocab_size,ctx=ctx)
    for epoch in range(num_epochs):
        train_loss = 0.0
        data_iter = data_iter_random(num_steps,batch_size,idx_corpus)
        num_example = 0.0
        for data,label in data_iter:
            data = one_hot(nd.array(data,ctx=ctx),vocab_size)
            H = nd.zeros(shape=(batch_size,hidden_dim),ctx=ctx)
            label = nd.array(label,ctx=ctx)
            with mx.autograd.record():
                outputs,H = rnn_network(data,H,*params)
                label = label.T.reshape((-1,))
                outputs = nd.concat(*outputs, dim=0)
                myloss=loss(outputs,label)
            myloss.backward()
            grad_clipping(params,5,ctx=ctx)
            util.update(params,learning_rate)
            train_loss += nd.sum(myloss).asscalar()
            num_example += myloss.size
        print('Epoch：%d  Prelexity: %f'%(epoch,np.exp(train_loss/num_example)))
        #print(np.exp(train_loss/num_example))

if __name__=='__main__':
    ctx = util.try_gpu()
    content,idx_to_char,char_to_idx,idx_corpus,vocab_size = load_data()
    #construct rnn
    hidden_dim = 128
    output_dim = vocab_size
    input_dim = vocab_size
    batch_size  = 56
    learning_rate = 0.1
    num_epochs = 50
    params = get_params(input_dim,hidden_dim,output_dim)
    num_steps=10
    train_and_predict(rnn_network = rnn,get_params = get_params,num_steps=num_steps,idx_corpus = idx_corpus,num_epochs=num_epochs,batch_size=batch_size,vocab_size=vocab_size,hidden_dim=hidden_dim,learning_rate=learning_rate,ctx=ctx)

