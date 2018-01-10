# 使用mxnet搭建一个识别汉字是否颠倒的分类器，想法和一些代码来自于知乎专栏：https://zhuanlan.zhihu.com/p/25297378

## 生成训练和测试数据

+ generate_img.py

- 虽然将这个代码适配到了Python3中，但是还是有问题，经常会出现uncodedecode error，是因为随机生成的unicode没有对应的汉字编码。这个情况我记得在python2情况下没有现在这么严重，现在循环100000只会有十几二十次能成功生成汉字。

- 执行该代码的python2版本比python3版本效果好很多


## 读取数据

这里使用了mxnet的高级语法: image.ImageIter()，不过该方法有一个

**segmentfault错误**

https://github.com/apache/incubator-mxnet/issues/5407



**import cv2** 错误
```python
pip install opencv-python
```
https://stackoverflow.com/questions/32389599/anaconda-importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-o


## 构建卷积网络并初始化
nn.conv2D的参数设置

图片经过卷积层过滤后的大小变化，假设长为x,padding 大小为p ,kernel_size 为 k*k,stride 为 s,那么卷积输出的长为 (x+2\*p-k)/s


## 参数初始化有什么讲究
- 最一般的初始化,gluon的API
```python
net.initialize()
```
## 怎么查看网络中各层的输出大小
### 可以通过在jupyter notbook中使用 ??这个语法查看各个类的具体信息。
- 直接输出网络会给出各个网络层的信息，比如kernel size ,padding等，但不会给出各个层的输出
```python
print(net)
```
- 查看具体某一层的参数信息
```python
print(net[0].params)

'''sequential2_conv0_ (
  Parameter sequential2_conv0_weight (shape=(5, 3, 2, 2), dtype=<class 'numpy.float32'>)
  Parameter sequential2_conv0_bias (shape=(5,), dtype=<class 'numpy.float32'>)
)'''

``` 

## 怎么在GPU上运行程序



## 怎么用symbol
- 使用mxnet.symbol api。
- module api 是用来对symbol搭建的网络进行训练的。
- net.infer_shape 用来推断各个层的变量的shape

- 我服了，module模块使用的是mx.io.NDArrayIter ，而我的图片用的是image.ImageIter, 试了一下，ImageIter不能用在module中。可以重写一个自己的iter。
- 我晕，最后一层必须要是softmax,不然会出现keyword error 错误

### 怎么在mxnet中训练多任务网络，不同任务的损失相加并优化
使用symbol可以实现这个功能
https://github.com/apache/incubator-mxnet/blob/master/example/multi-task/example_multi_task.py



## checkpoint 的用法


## 画出loss或acc的曲线变化