#%%

from mxnet import ndarray as nd
from mxnet import gluon
from util import accuracy,evaluate_accuracy,SGD



def pure_batch_norm(x,gamma,beta,eps=0):
    assert len(x.shape) in (2,4)
    if len(x.shape) == 2:
        mean = x.mean(axis=0)
        variance = ((x-mean)**2).mean(axis=0)
    else:
        mean = x.mean(axis = (0,2,3),keepdims = True)
        variance = ((x-mean)**2).mean(axis=(0,2,3),keepdims = True)

    x_hat =  (x-mean)/nd.sqrt(variance+eps)
    #return gamma.reshape(mean.shape) * x_hat + beta.reshape(mean.shape)
    return x_hat

        
    
#test 一维卷积
#a = nd.arange(6).reshape((3,2))
#print(a)
#print(pure_batch_norm(a, gamma = nd.array([1,1]), beta = nd.array([0,0])))
#
#twodim = nd.arange(12).reshape((1,2,3,2))
#print(twodim)
#print(pure_batch_norm(twodim,gamma = nd.array([1,1]), beta = nd.array([0,0])))

def batch_norm(x,gamma,beta,is_training,moving_mean,moving_variance,eps = 1e-5,moving_momentum = 0.9): 
    assert len(x.shape) in (2,4)
    if len(x.shape) == 2:
        mean= x.mean(axis=0)
        variance = ((x-mean)**2).mean(axis=0)
    else:
        mean = x.mean(axis = (0,2,3),keepdims = True)
        variance = ((x-mean)**2).mean(axis=(0,2,3),keepdims = True)
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)

    if is_training:
        x_hat = ((x-mean))/nd.sqrt(variance+eps)
        moving_mean = moving_mean*moving_momentum + (1-moving_momentum)*mean
        moving_variance = moving_variance*moving_momentum + (1-moving_momentum)*variance
    else:
        x_hat = (x-moving_mean)/nd.sqrt(moving_variance+eps)
    return gamma.reshape(mean.shape)*x_hat + beta.reshape(mean.shape)

# 构建网络
# 怎样对下面的CNN网络添加一个batch normalization 层
# BN 添加在卷积操作之后，激活函数之前
import mxnet as mx
try:
    ctx = mx.gpu()
    _=nd.array([1],ctx=ctx)
except:
    ctx = mx.cpu()

print(ctx)

weight_scale = .01
# output channels:20, input channel:1, kernel size: 5*5
w1 = nd.random_normal(shape=(20,1,5,5),scale = weight_scale,ctx= ctx)
b1 = nd.zeros(w1.shape[0],ctx=ctx)
gamma1 = nd.random_normal(shape = 20,scale = weight_scale,ctx= ctx)
beta1 = nd.random_normal(shape = 20,scale = weight_scale,ctx= ctx)
moving_mean1 = nd.random_normal(shape = 20,scale = weight_scale,ctx= ctx)
moving_variance1 = nd.random_normal(shape = 20,scale = weight_scale,ctx= ctx)


w2 = nd.random_normal(shape=(50,20,3,3),scale = weight_scale,ctx=ctx)
b2 = nd.zeros(w2.shape[0],ctx = ctx)

gamma2 = nd.random_normal(shape = 50,scale = weight_scale,ctx= ctx)
beta2 = nd.random_normal(shape = 50,scale = weight_scale,ctx= ctx)

moving_mean2 = nd.zeros(shape=50,ctx=ctx)
moving_variance2 = nd.zeros(shape = 50, ctx = ctx)
#moving_mean2 = nd.random_normal(shape = 50,scale = weight_scale,ctx= ctx)
#moving_variance2 = nd.random_normal(shape = 50,scale = weight_scale,ctx= ctx)


#1250怎么计算的？
w3 = nd.random_normal(shape=(1250,128),scale = weight_scale,ctx = ctx)
b3 = nd.zeros(w3.shape[1],ctx = ctx)

#gamma3 = nd.random_normal(shape = 128,scale = weight_scale,ctx= ctx)
#beta3 = nd.random_normal(shape = 128,scale = weight_scale,ctx= ctx)
#moving_mean3 = nd.random_normal(shape = 128,scale = weight_scale,ctx= ctx)
#moving_variance3 = nd.random_normal(shape = 128,scale = weight_scale,ctx= ctx)



w4 = nd.random_normal(shape=(w3.shape[1],10),scale = weight_scale,ctx = ctx)
b4 = nd.zeros(w4.shape[1],ctx = ctx)

params = [w1,b1,w2,b2,w3,b3,w4,b4]
for param in params:    
    param.attach_grad()

#%%
def net(x,verbose=False):
    x = x.as_in_context(w1.context)
    h1_conv = nd.Convolution(x,w1,b1,kernel = w1.shape[2:],num_filter = w1.shape[0])
    h1_bn = batch_norm(h1_conv,gamma1,beta1,is_training = True,moving_mean = moving_mean1,moving_variance=moving_variance1)
    h1_activation = nd.relu(h1_bn)
    h1 = nd.Pooling(data = h1_activation,pool_type='max',kernel=(2,2),stride = (2,2))

    h2_conv = nd.Convolution(h1,w2,b2,kernel = w2.shape[2:],num_filter = w2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2_pooling = nd.Pooling(data = h2_activation,pool_type='max',kernel=(2,2),stride = (2,2))
    h2 = nd.flatten(h2_pooling)

    h3_linear = nd.dot(h2,w3) + b3
    h3 = nd.relu(h3_linear)

    h4_linear = nd.dot(h3,w4) + b4
    
    if verbose:
        print('1st conv block:',h1.shape)
        print('2nd conv block:',h2.shape)
        print('1st dense block:',h3.shape)
        print('2nd dense block:',h4_linear.shape)
    return h4_linear


epoches = 5
softmax_crossentropy = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = .1

batch_size = 256
import util
from mxnet import autograd
train_data,test_data =  util.load_data_fashion_mnist(batch_size)

for epoch in range(epoches):
    train_acc = 0.0
    train_loss = 0.0

    for data,label  in train_data:
        data = data.reshape((-1,1,28,28))
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        #data reshape and as_in_context
        #label = label.as_in_context()
        with autograd.record():
            output = net(data)
            loss = softmax_crossentropy(output,label)

        loss.backward()
        #因为bakcward中各个参数的梯度的值是其累加值。
        SGD(params,learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar() 
        train_acc += accuracy(output,label)
    test_acc  = evaluate_accuracy(test_data,net,ctx)
    print("epoch: %d, train_loss:%f, train_acc:%f, test_acc:%f"%(epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))

