# learn_mxnet
This is the code I write to learn the deep learning framework mxnet

- 关于 mxnet中的name_scope

```python
with self.name_scope():
    self.dense0 = nn.Dense(10)
```
相当于给name_scope下的参数名加了一个默认前缀，是为了参数名字的独一无二性。一般每一层都应该定义自己的name_scope。

## RNN
## CNN
## GAN
## REINFORCEMENT LEARNING