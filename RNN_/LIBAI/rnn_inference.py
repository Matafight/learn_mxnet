#_*_coding:utf-8_*_
'''
inference for rnn
'''
import mxnet as mx
from hybrid_network import rnnModel,get_parameters,load_data,try_gpu



context = try_gpu()
#load existing model
batch_size,embed_dim,num_steps,num_hidden = get_parameters()
data_iter,vocab_size = load_data(batch_size,num_steps,context)
#mod = rnnModel(batch_size,vocab_size,embed_dim,num_hidden,num_steps)

#given first several words to predict the next words


#导入checkpoint 里的模型
model_prefix='poem_rnn'
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 5)
mod = mx.mod.Module(sym)
# assign the loaded parameters to the module
mod.bind(data_shapes = data_iter.provide_data,label_shapes=data_iter.provide_label)
mod.set_params(arg_params, aux_params)
checkpoint = mx.callback.do_checkpoint(model_prefix)
print('start training...')
mod.fit(data_iter,
        num_epoch=10,
        arg_params=arg_params,
        aux_params=aux_params,
        epoch_end_callback=checkpoint,
        begin_epoch=5)