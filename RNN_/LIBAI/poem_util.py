#_*_ coding:utf-8_*_

'''
1. using RNN to train poems
2. support checkpoint support
3. support specify the first word of poem when doing inference
4. need begin and stop token for each sentences, why?
5. try using <UNK> token?
6. only need to unpack the lastone by using split(':')
7. reshape((,-1)) 's method
'''
import collections as cl
import random
import mxnet.ndarray as nd
from mxnet.gluon import nn,autograd
from mxnet import gluon
import numpy as np
import mxnet as mx

#没有考虑停止词
def transform_data(path,num_steps):   
    poems = []
    with open(path,encoding='utf-8') as fh:
        lines = fh.readlines()
        for line in lines[:500]:
            *title,content = line.strip().split(':')
            content.replace(' ','')
            #content may be empty
            if (len(content)>num_steps or len(content)==0):
                continue
            poems.append(content)

    #逗号怎么处理呢？
    lines = sorted(poems,key = lambda x:len(x))

    words = []
    for line in lines:
        #可能包含逗号
        words += [ word for word in line]

    #generate word_int_map

    word_counter = cl.Counter(words)
    #dict type
    words_list = sorted(word_counter.items(),key = lambda x:-x[1])
    #zip的用法？
    unique_word,_=zip(*words_list)
    #generate dict
    word_to_int = dict(zip(unique_word,range(len(unique_word))))
    int_to_word = unique_word

    # map original sentences to ints
    #corpus_vec=[]
    #for line in poems:
    #    cur_vec = [word_to_int[x] for x in line]
    #    corpus_vec.append(cur_vec)
    #return corpus_vec,word_to_int

    #here is wrong , I need correct this later
    poems_vec = [list(map(lambda l:word_to_int.get(l,0),poem)) for poem in poems]
    #poems_vec = [list(map(lambda l:word_to_int.get(l,word_to_int.get(' ')),poem)) for poem in poems]
    return poems_vec,word_to_int,int_to_word

def train_test_split(poems_vec):
    # 70% for training and 30% for testing 
    corpus_len = len(poems_vec)
    random.shuffle(poems_vec)
    training_pos = int(corpus_len*0.7)
    training_vec = poems_vec[0:training_pos]
    testing_vec = poems_vec[training_pos:]
    return training_vec,testing_vec
    

def generate_batch(poems_vec,word_to_int,batch_size,ctx=mx.cpu()):
    num_batch = len(poems_vec)//batch_size
    idx_batch = list(range(num_batch))
    random.shuffle(idx_batch)
    #num_steps
    max_len = max(map(len,poems_vec)) 
    for idx in idx_batch:
        start_pos = idx*batch_size
        end_pos = (idx+1)*batch_size
        #batch_data = poems_vec[start_pos:end_pos]
        batch_data = nd.full((batch_size,max_len),0,ctx=ctx)
        for row,line in enumerate(poems_vec[start_pos:end_pos]):
            temp = nd.array(line)
            batch_data[row,0:len(line)] = temp
        
        #generate label
        batch_label=batch_data.copy()
        batch_label[:,0:batch_label.shape[1]-1] = batch_data[:,1:batch_label.shape[1]]
        #batch_size * num_steps
        yield (batch_data,batch_label)



#define network

            

if __name__=='__main__':
    poem_vec,word2int = transform_data('../input/poems.txt')
    data_iter = generate_batch(poem_vec,word2int,batch_size=32)
    for i,(data,label) in enumerate(data_iter):
        print(i)
        print(data.shape)
        print(label.shape)
    
 