#_*_coding:utf-8_*_
'''
static graph
'''
from mxnet import rnn
import mxnet as mx
import mxnet.ndarray as nd
from poem_util import transform_data, generate_batch, train_test_split
from data import *
# parameters
batch_size = 128
embed_dim = 200
num_steps = 90
num_hidden = 100
# load data
def try_gpu():
    try:
        ctx=mx.gpu()
        _ = nd.array([0],ctx=ctx)
    except:
        ctx=mx.cpu()
    return ctx

#data
corpus_vec,word_to_int = transform_data('../input/poems.txt',num_steps)
vocab_size = len(word_to_int)
data_iter = CustomIter(corpus_vec,batch_size,num_steps)
#model

def poem_rnn(batch_size,vocab_size,embed_dim,num_hidden,num_steps,ctx=mx.cpu()):
    seq_input = mx.symbol.Variable('data',ctx=ctx)

    #seq_inupt = mx.symbol.Reshape(seq_input,shape=(batch_size,num_steps))
    embedded_seq = mx.symbol.Embedding(data=seq_input, 
                                       input_dim=vocab_size, 
                                       output_dim=embed_dim)
    #weird!why reshape seq_input before embedding not working?                                     
    #prove this reshape function reshape the input in the right direction
    embedded_seq = mx.sym.Reshape(embedded_seq,shape=(batch_size,num_steps,embed_dim))
    lstm_cell = mx.rnn.LSTMCell(num_hidden=num_hidden)
    #NTC means batch_size, num_steps,input_dimensions
    #outputs is merged into a single symbol of shape (batch_size,num_steps,hidden_dim)
    outputs, _ = lstm_cell.unroll(length=num_steps, 
                                       inputs=embedded_seq, 
                                       layout='NTC', 
                                       merge_outputs=True)
    #decoder
    pred = mx.sym.Reshape(outputs,shape=(-1,num_hidden))
    pred = mx.sym.FullyConnected(data = pred,num_hidden=vocab_size,name='pred')
    pred = mx.sym.Reshape(pred,shape=(-1,vocab_size),name='pred1')
    pred = mx.sym.SoftmaxOutput(data = pred,name='softmax')
    return pred

#def cross_entropy_loss(pred):
#    label = mx.sym.Variable('label')
#    #input label's shape (batch_size,num_steps),after reshape (batch_size*num_steps,) 
#    label = mx.sym.Reshape(label,shape=(-1,))
#    logits = mx.sym.log_softmax(pred,axis=-1)
#    loss =  -mx.sym.pick(logits,label,axis=-1,keepdims = True)
#    loss = mx.sym.mean(loss,axis=0,exclude=True)
#    return mx.sym.make_loss(loss,name='nll')

# training network

output = poem_rnn(batch_size,vocab_size,embed_dim,num_hidden,num_steps)
mod = mx.mod.Module(output,data_names=["data"],label_names=["softmax_label"])
mod.bind(data_shapes = data_iter.provide_data,label_shapes=data_iter.provide_label)
mod.init_params(initializer=mx.init.Xavier())
mod.init_optimizer(optimizer='sgd',optimizer_params=(('learning_rate',0.1),))

# loss.backward()
metric = mx.metric.create('acc')
num_epoch = 10
for epoch in range(num_epoch):
    data_iter.reset()
    metric.reset()
    for batch in data_iter:
        mod.forward(batch,is_train=True)
        print(mod.output_shapes)
        mod.update_metric(metric,batch.label)
        mod.backward()
        mod.update()
    print('epoch %d train acc: %s' % (epoch,metric.get()))
