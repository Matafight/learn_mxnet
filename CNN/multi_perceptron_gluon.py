

#%%
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import util

batch_size = 256
train_data,test_data =  util.load_data_fashion_mnist(batch_size)

num_inputs = 28*28
num_outputs = 10
num_hiddens = 256

net = gluon.nn.Sequential()


print("part1")
#%%
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256,activation = 'relu'))
    net.add(gluon.nn.Dense(10))
net.initialize()


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.5})

epoches = 5
for epoch in range(epoches):
    train_loss = 0
    train_acc = 0
    for data,label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += util.accuracy(output,label)
    test_acc = util.evaluate_accuracy(test_data,net)
    print("epoch:%d , train_loss:%f, train_acc:%f, test_acc:%f"%(epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))