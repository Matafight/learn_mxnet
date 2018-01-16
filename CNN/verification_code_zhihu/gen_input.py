#_*_coding:utf-8_*_
from mxnet import image
import os

def extract_label(name):
    ret = name.split('.')[0].split('_')[1]
    if(len(ret)==0):
        return [0,0]
    elif(len(ret)==2):
        return [1,1]
    elif(len(ret)==1 and ret=='0'):
        return [1,0]
    elif(len(ret)==1 and ret=='1'):
        return [0,1]

#自己写一个生成path_imglist的函数。
#index    labels    path
def gen_imglist(data_dir,data_name):
    files = os.listdir(data_dir)
    indexes = len(files)
    with open('./'+data_name+'.lst','w') as fh:
        for ind,file in enumerate(files):
            line = str(ind)+'\t'
            label = extract_label(file)
            #line += str(label[0]) +'\t'+ file+'\n'
            line += str(label[0]) +'\t'+str(label[1]) +'\t'+ file+'\n'
            fh.write(line)
    return indexes
            
            
def gen_input(data_dir,data_name):
    num_files = gen_imglist(data_dir,data_name)     
    data_iter = image.ImageIter(batch_size=num_files,shuffle = True,data_shape=(3,28,56),label_width=2,path_imglist='./'+data_name+'.lst',path_root = data_dir)
    return data_iter
