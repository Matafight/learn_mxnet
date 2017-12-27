

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

def SGD(params,lr):
    for param in params:
        param[:] = param -lr*param.grad
        
        
        
def accuracy(output,label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()


def evaluate_accuracy(data_iter,net,ctx):
    acc = 0
    for data,label in data_iter:
        data = data.reshape((-1,1,28,28))
        output = net(data.as_in_context(ctx))
        label = label.as_in_context(ctx)
        acc += accuracy(output,label)
    return acc/len(data_iter)


def transform(data,label):
    return data.astype('float32')/255,label.astype('float32')


def load_data_fashion_mnist(batch_size):
    mnist_train = gluon.data.vision.FashionMNIST(train = True,transform = transform)
    minst_test = gluon.data.vision.FashionMNIST(train = False,transform = transform)
    train_data = gluon.data.DataLoader(mnist_train,batch_size,shuffle= True)
    test_data = gluon.data.DataLoader(minst_test,batch_size,shuffle = True)
    #需要下载数据
    return train_data,test_data
