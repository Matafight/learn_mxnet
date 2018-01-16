#%%
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import util
import time
batch_size = 256
train_data,test_data =  util.load_data_fashion_mnist(batch_size)
#需要reshape 一下data

'''
w = nd.arange(4).reshape((1,1,2,2))
b = nd.array([1])
data = nd.arange(9).reshape((1,1,3,3))
out = nd.Convolution(data,w,b,kernel = w.shape[2:],num_filter = w.shape[0])

print('input:',data,'\n\nweight',w,'\nb',b,'\nout',out)

#%%
out = nd.Convolution(data,w,b,kernel = w.shape[2:],num_filter = w.shape[0],stride = (2,2),pad = (1,1))
print('input:',data,'\n\nweight',w,'\nb',b,'\nout',out)


#%% 
w = nd.arange(8).reshape((1,2,2,2))
b = nd.array([1])
data = nd.arange(18).reshape((1,2,3,3))

out = nd.Convolution(data,w,b,kernel = w.shape[2:],num_filter = w.shape[0],stride = (2,2),pad = (1,1))
print('input:',data,'\n\nweight',w,'\nb',b,'\nout',out)


#%%
#input  two channels
w = nd.arange(16).reshape((2,2,2,2))
b = nd.array([2,2])
data = nd.arange(18).reshape((1,2,3,3))
out = nd.Convolution(data,w,b,kernel = w.shape[2:],num_filter = w.shape[0],stride = (2,2),pad = (1,1))
print('input:',data,'\n\nweight',w,'\nb',b,'\nout',out)


#%%
#Pooling
data = nd.arange(18).reshape((1,2,3,3))
max_pool = nd.Pooling(data = data,pool_type = 'max',kernel=(2,2))
avg_pool = nd.Pooling(data = data,pool_type = 'avg',kernel = (2,2))
print('data:',data,'\n max_pool:',max_pool,'\navg_pool:',avg_pool)

'''
#%%
#定义模型
import mxnet as mx
try:
    ctx = mx.gpu()
    _ = nd.array([1,2],ctx=ctx)
except:
    ctx = mx.cpu()
ctx
print(ctx)
#使用LeNet

weight_scale = .01
w1 = nd.random_normal(shape=(20,1,5,5),scale = weight_scale,ctx= ctx)
b1 = nd.zeros(w1.shape[0],ctx=ctx)

w2 = nd.random_normal(shape=(50,20,3,3),scale = weight_scale,ctx=ctx)
b2 = nd.zeros(w2.shape[0],ctx = ctx)

#1250怎么计算的？
w3 = nd.random_normal(shape=(1250,128),scale = weight_scale,ctx = ctx)
b3 = nd.zeros(w3.shape[1],ctx = ctx)

w4 = nd.random_normal(shape=(w3.shape[1],10),scale = weight_scale,ctx = ctx)
b4 = nd.zeros(w4.shape[1],ctx = ctx)

params = [w1,b1,w2,b2,w3,b3,w4,b4]
for param in params:    
    param.attach_grad()


def net(x,verbose=False):
    x = x.as_in_context(w1.context)
    h1_conv = nd.Convolution(x,w1,b1,kernel = w1.shape[2:],num_filter = w1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data = h1_activation,pool_type='max',kernel=(2,2),stride = (2,2))

    h2_conv = nd.Convolution(h1,w2,b2,kernel = w2.shape[2:],num_filter = w2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2_pooling = nd.Pooling(data = h2_activation,pool_type='max',kernel=(2,2),stride = (2,2))
    h2 = nd.flatten(h2_pooling)

    h3_linear = nd.dot(h2,w3) + b3
    h3 = nd.relu(h3_linear)

    h4_linear = nd.dot(h3,w4) + b4
    
    if verbose:
        print('1 st conv block:',h1.shape)
        print('2nd conv block:',h2.shape)
        print('1st dense block:',h3.shape)
        print('2nd dense block:',h4_linear.shape)
    return h4_linear

#%%
#训练

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from util import SGD,accuracy,evaluate_accuracy


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .2

epoches = 5

start = time.clock()
for epoch in range(epoches):
    train_loss = 0.0
    train_acc = 0.0
    for data,label in train_data:
        data = data.reshape((-1,1,28,28))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        SGD(params,learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net,ctx)
    print("Epoch %d. LOss:%f,train acc:%f,test acc:%f"%(epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))

elapse = time.clock()-start
print("total cost time : %f"%elapse)


   

    

    




