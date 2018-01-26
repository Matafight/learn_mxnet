## mxnet gluon 版本的自动作诗机器人,来源于 github项目:https://github.com/jinfagang/tensorflow_poems
### 读取数据
#### 详细介绍数据在网络中的传递过程以及数据维度的转变
这是我用gluon写得第一个关于RNN的程序，network.py已经可以训练了，这个代码的难点在于理解数据的输入方式，以及要清楚地知道RNN的原理和数据在网络中流动时的维度变化。

- poem_util.py 读取数据并且生成数据迭代器，数据迭代器输出每个batch的data和label，其中**data.shape = (batch_size,num_steps), label.shape = (batch_size,num_steps)**。

- network.py中的这段代码值得关注
```python
  for ibatch,(data,target) in enumerate(train_iter):
        data = data.T
        target = target.T.reshape((-1,))
```
    因为输入的是文本数据，需要将每个汉字转换为词向量，所以需要调用Embedding方法，假设nn.Embedding的输入shape为(batch_size,num_steps)，则Embedding的输出shape为(batch_size,num_steps,embed_dim)。由于rnn.RNN(GRN,LSTM)等方法默认接受输入的layout为"TNC"，就是(num_steps,batch_size,input_dim)的形式，所以需要先对data转置。至于target为什么要转置呢？这涉及到reshape()方法的作用原理。


RNN的输出output的shape为(num_steps,batch_size,hidden_dim)
```python
output,state = self.rnn(emb,state)
```
这个output表示了 **num_steps\*batch_size**个字符的隐状态，需要预测这么多个字符。为了预测字符，需要将output输入到一层DenseLayer中
```python
self.decoder = nn.Dense(vocab_size,in_units=hidden_dim)
...
decoded = self.decoder(output.reshape((-1,self.hidden_dim)))
```
其中output.reshape((-1,self.hidden_dim))的意思是将outputreshape成两维，其中第二个维度为hidden_dim，第一个维度根据原始output的维度自动推算得出，由于原始output的shape为(num_steps,batch_size,hidden_dim),所有reshape之后的维度为(num_steps*batch_size,hidden_dim)。那么原来的(num_steps,batch_size)这两个维度是按照什么方向拼接成一维的呢?
实验验证一下，原始数据shape=(3,4,5)

![](http://7xiegr.com1.z0.glb.clouddn.com/reshape1.PNG)

reshape((-1,5))之后为:

![](http://7xiegr.com1.z0.glb.clouddn.com/reshape2.PNG)

所以，output.reshape((-1,self.hidden_dim))之后的output是这样的:

![](http://7xiegr.com1.z0.glb.clouddn.com/reshape3.png)

由于在train_and_eval函数中计算损失时是num_steps*batch_size这么多个样本一起计算而没有使用循环，所以对label也要reshape，将(batch_size,num_steps)的label转置成(num_steps,batch_size)。从而使output的每一行与label的每一行一一对应，计算loss时不会算错。
![](http://7xiegr.com1.z0.glb.clouddn.com/reshape4.png)





#### 如何加速训练-hybridize

- 因为gluon支持动态图和静态图的转换，动态图的优点是方便调试，静态图的优点是运行效率高。通过继承HybridBlock或HybridSequential来定义网络,新的网络默认是动态图，可以在动态图的情况下debug，当确认没有错误时，调用hybridize()方法将网络转换为静态图。

- 貌似GRU、RNN、LSTM等并不支持hybridize，GRUCell支持hybridize。但是使用GRUCell就意味着需要对num_steps进行循环，因为一个cell只能处理一个time\_step的数据。

- 如果想构建基于RNN的静态图，而又不使用RNNCell，应该使用mxnet.RNNCell相关API定义网络 https://mxnet.incubator.apache.org/versions/master/api/python/symbol/rnn.html#mxnet.rnn.GRUCell。

- rnn_unroll_module.py 的代码就是使用mx.sym定义的静态网络，需要用mx.mod.Module wrap起来，由于在定义网络时time_steps这个参数是固定的，这样在训练时采用固定的time_steps训练。当做预测时，无法使用给定的一个word inference出后续的word。只能给定与time_steps相同长度的输入，由module计算其输出。这对于RNN来说是非常大的限制。

- 为了克服上面的使用mx.mod.Module模块带来的问题，可以采用gluon API下的RNNCell，虽然一个Cell只能处理一个time的input，但是可以通过循环的方式来训练。



参见:
1. https://discuss.gluon.ai/t/topic/1828/7
2. https://mxnet.incubator.apache.org/api/python/gluon/rnn.html

#### Block和Sequential之间是什么关系?
Block是mxnet中所有layer的基类，Sequential的作用是将各个Block拼接在一起。
- Block的使用方法:https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.Block
- Sequential的使用方法:https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.nn.Sequential


### 训练网络
#### 分为以下几种方式
- 使用mx.mod.Module 的fit方法
- 也是使用mx.mod.Module，不过使用其step by step的训练方式，参见rnn_unroll_module.py中的相关函数。
- 采用gluon的训练方式，清晰地写出每一步代码，参见 network.py中的训练函数
### 自动作诗(推理阶段)
- mx.mod.Module模块不适合给定一个单词预测接下来多个单词的情形。


