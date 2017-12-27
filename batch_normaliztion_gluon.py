#%%
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import mxnet as mx
import util


batch_size = 20
train_data,test_data =  util.load_data_fashion_mnist(batch_size)



from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(channels = 20,kernel_size = 5))
    net.add(nn.BatchNorm(axis = 1))
    net.add(nn.Activation(activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 2,strides = 2))

    net.add(nn.Conv2D(channels = 50,kernel_size = 3))
    net.add(nn.BatchNorm(axis = 1))
    net.add(nn.Activation(activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 2,strides = 2))

    #dense 
    net.add(nn.Flatten())
    net.add(nn.Dense(128,activation = 'relu'))
    net.add(nn.Dense(10))


try:
    ctx = mx.gpu()
    _ = nd.array([1],ctx = ctx)
except:
    ctx = mx.cpu()
print(ctx)

#%%
net.initialize(ctx= ctx)
trainer = gluon.Trainer( net.collect_params(),'sgd',{'learning_rate':0.1})

crossentropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
epoches = 5
learning_rate = 0.1
for epoch in range(epoches):
    train_loss = 0.0
    train_acc = 0.0
    for data,label in train_data:
        data = data.reshape((-1,1,28,28))
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = crossentropy_loss(output,label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += util.accuracy(output,label)
    test_acc  = util.evaluate_accuracy(test_data,net,ctx)
    print("epoch: %d, train_loss:%f, train_acc:%f, test_acc:%f"%(epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))
