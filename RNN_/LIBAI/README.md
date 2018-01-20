## mxnet gluon 版本的自动作诗机器人,来源于 github项目:https://github.com/jinfagang/tensorflow_poems
### 读取数据
#### 详细介绍数据在网络中的传递过程以及数据维度的转变
### 构建网络
#### 如何加速训练-hybridize

因为gluon支持动态图和静态图的转换，动态图的优点是方便调试，静态图的优点是运行效率高。通过继承HybridBlock或HybridSequential来定义网络,新的网络默认是动态图，可以在动态图的情况下debug，当确认没有错误时，调用hybridize()方法将网络转换为静态图。

貌似GRU、RNN、LSTM等并不支持hybridize，GRUCell支持hybridize。但是使用GRUCell就意味着需要对num_steps进行循环，因为一个cell只能处理一个time\_step的数据。

如果想构建基于RNN的静态图，而又不使用RNNCell，应该使用mxnet.symbol相关api定义网络。

参见:
1. https://discuss.gluon.ai/t/topic/1828/7
2. https://mxnet.incubator.apache.org/api/python/gluon/rnn.html
#### Block和Sequential之间是什么关系?
Block是mxnet中所有layer的基类，Sequential的作用是将各个Block拼接在一起。
- Block的使用方法:https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.Block
- Sequential的使用方法:https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.nn.Sequential


### 训练网络
### 自动作诗(推理阶段)

