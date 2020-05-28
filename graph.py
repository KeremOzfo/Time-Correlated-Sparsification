import numpy as np
import matplotlib.pyplot as plt
from os import *
import math
from itertools import cycle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

def special_adress():
    adress=[]
    labels = ['TimeC-Resnet-0.2','TimeC-Resnet-0.3','TimeC-Resnet-0.4','Single-MachineBenchmark-0.3']
    adress.append('Results/timeCorrelated/resnet/0.2')
    adress.append('Results/timeCorrelated/resnet/0.30')
    adress.append('Results/timeCorrelated/resnet/0.40')
    adress.append('Results/benchmark/resnet/0.3')
    return adress,labels

def compile_results(adress):
    results = None
    for i, dir in enumerate(listdir(adress)):
        vec = np.load(adress + '/'+dir)
        if i==0:
            results = vec/len(listdir(adress))
        else:
           results += vec/len(listdir(adress))
    return results

def cycle_graph_props(colors,markers,linestyles):
    randoms =[]
    randc = np.random.randint(0,len(colors))
    randm = np.random.randint(0,len(markers))
    randl = np.random.randint(0,len(linestyles))
    m = markers[randm]
    c = colors[randc]
    l = linestyles[randl]
    np.delete(colors,randc)
    np.delete(markers,randm)
    np.delete(linestyles,randl)
    print(colors,markers,linestyles)
    return c,m,l


def avgs(sets):
    avgs =[]
    for set in sets:
        avg = np.zeros_like(set[0])
        avgs.append(avg)
    return avgs

def graph(data, legends,interval):
    marker = ['s', 'v', '+', 'o', '*']
    color = ['r', 'c', 'b', 'g','y','b','k']
    linestyle =['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    colorcycler = cycle(color)
    markercycler = cycle(marker)
    for d,legend in zip(data,legends):
        x_axis = []
        l = next(linecycler)
        m = next(markercycler)
        c = next(colorcycler)
        for i in range(0,len(d)):
            x_axis.append(i*interval)
        plt.plot(x_axis,d, marker= m ,linestyle = l ,markersize=2, label=legend)
    #plt.axis([0, 30,70 ,90])
    #plt.axis([145,155,88,92])
    #plt.axis([280, 300, 87, 95])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('LayerWise- Time Correlated')
    plt.legend()
    plt.show()



def concateresults(dirsets):
    all_results =[]
    for set in dirsets:
        all_results.append(compile_results(set))
    return all_results


loc = 'Results/'
types = ['benchmark','timeCorrelated','topk']
NNs = ['simplecifar']

locations = []
labels =[]
for tpye in types:
    for nn in NNs:
        locations.append(loc + tpye +'/'+nn)
        labels.append(tpye +'--'+ nn)

intervels = 1
labels = special_adress()[1]
results = concateresults(special_adress()[0])
#results = concateresults(locations)
graph(results,labels,intervels)
#data,legends = compile_results(loc)
#graph(data,labels,intervels)