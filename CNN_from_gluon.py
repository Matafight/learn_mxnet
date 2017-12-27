
#%%
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import util
import time

batch_size = 256
train_data,test_data =  util.load_data_fashion_mnist(batch_size)

from mxnet.gluon import nn

net = nn.Sequential()

with net.name_scope():
    net.add(nn.Conv2D(channels = 20, kernel_size = 5, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2,strides=2))
    net.add(nn.Conv2D(channels=50,kernel_size = 3, activation= 'relu'))
    net.add(nn.MaxPool2D(pool_size=2,strides=2))
    net.add(nn.Flatten())
    net.add(nn.Dense(128,activation='relu'))
    net.add(nn.Dense(10))


#将权重初始化在cpu上

import mxnet as mx
from util import SGD,accuracy,evaluate_accuracy

try:
    ctx = mx.gpu()
    _ = nd.array([1,2],ctx = ctx)
except:
    ctx = mx.cpu()

print(ctx)


net.initialize(ctx = ctx)
#获取数据后训练

print("part1 finished")

#%%
epoches = 5

trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.5})
softmax_crossentropy = gluon.loss.SoftmaxCrossEntropyLoss()
start = time.clock()
for epoch in range(epoches):
    train_acc = 0.0
    train_loss = 0.0

    for data,label in train_data:
        data = data.reshape((-1,1,28,28))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_crossentropy(output,label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net,ctx)
    print("epoch:%d, train_loss:%f, train_acc:%f, test_acc:%f"%(epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))
elapse = time.clock()-start
print('elapsing time %f'%elapse)