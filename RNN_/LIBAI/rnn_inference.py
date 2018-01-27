#_*_coding:utf-8_*_
'''
inference for rnn
'''
import mxnet as mx
from rnn_unroll_module import rnnModel,get_parameters,load_data,try_gpu
import time




def load_existing_model_stepbystep(start_epoch):
    context = try_gpu()
    batch_size,embed_dim,num_steps,num_hidden = get_parameters()
    data_iter,vocab_size = load_data(batch_size,num_steps,context)
    prefix = 'stepbystep_rnn'
    mod = mx.mod.Module.load(prefix=prefix,epoch=start_epoch,load_optimizer_states=True,data_names=['data'],label_names=['softmax_label'],context=context)
    mod.bind(data_shapes = data_iter.provide_data,label_shapes=data_iter.provide_label)
    mod.init_optimizer(optimizer='sgd',optimizer_params=(('learning_rate',0.1),))
    metric = mx.metric.create('acc')

    num_epoch = 1
    start = time.time()
    model_prefix='stepbystep_rnn'
    for epoch in range(start_epoch+1,start_epoch+1+num_epoch):
        data_iter.reset()
        metric.reset()
        for batch in data_iter:
            mod.forward(batch,is_train=True)
            mod.update_metric(metric,batch.label)
            mod.backward()
            mod.update()
        if epoch %1 ==0:
            mod.save_checkpoint(prefix=model_prefix,epoch = epoch,save_optimizer_states=True)
        print('epoch %d train acc: %s' % (epoch,metric.get()))
        print('elapse time:%s seconds'%(time.time()-start))
    return mod



def load_existing_model_integrated():
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
    return mod

#the rnn constructed from mx.mod.Module seems not easy to inference from only a word,how? seems not possible, but we 
#can input a sentence with lenght as num_steps,then using the slide windows to continue produce the output words, this idea is feasible
#update on: 1/27 2018
#However, the module has already bind to a specific input shape using the data.provide_data and data.provide_label, and since the data.provide_data provide training data'shape related to the batch size, so , it seems that when doing referene ,we also need to provide test data with the same shape with the shape provided by data.provide_data methd. Thus, using mx.mod.Module to inference the following words given only one or several words seem not possible
def inference(mod):
    #predict 
    # mod.predict(data_iter)
    # construct start_vec as a dataiter
    test_iter = mx.io.NDArrayIter(mx.nd.ones(90), batch_size=1)

    test_output = mod.predict(test_iter)

if __name__ == '__main__':
    start_epoch = 2
    mod = load_existing_model_stepbystep(start_epoch)
    