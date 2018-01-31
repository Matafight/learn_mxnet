#_*_coding:utf-8_*_
'''
This is a collection of many reuseable functions
'''
import seaborn as sen
import matplotlib.pyplot as plt
#draw curve w.r.t iterations
def draw_curve(vec1,legendfir = 'First',xlabel='x',ylabel='y',vec2=None,legendsec='Second',save_path='.'):
    ax = sen.tsplot(vec1,color='r',condition=legendfir,legend=True)
    if vec2!=None:
        ax2 = sen.tsplot(vec2,color='b',condition=legendsec,legend=True)
    ax.set(xlabel=xlabel,ylabel=ylabel)
    plt.savefig(save_path+'/test.png')
     
