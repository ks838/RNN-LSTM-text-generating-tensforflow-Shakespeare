
'''
@ Author:  Kai Song, ks838@cam.ac.uk

@ Notes:   What does this small project do?
           1. This is the character statistics part.
'''

import time
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
from os import listdir
from os.path import isfile, join
import codecs
#import plotly.plotly as py
#import plotly.graph_objs as go

start_time = time.time()

print(__doc__)
x = range(26)
u_all = np.zeros(26)
alp = 'abcdefghijklmnopqrstuvwxyz'
#print('A'=='a')#False

path_shake = './complete_works/'
all_files = [f.replace('.txt','') for f in listdir(path_shake) if isfile(join(path_shake, f))]


n_files = len(all_files)

#filename = 'alls_well_ends_well.txt'

for i in range(2):
    file_name = './complete_works/'+ all_files[i]+'.txt'
    u = np.zeros(26)
    lines = codecs.open(file_name, "r",encoding='utf-8', errors='ignore').readlines()
    for y in range(0,26):
        for line in lines:
            for char in line.lower():
                if char == alp[y]:
                    u[y] += 1
                    u_all[y] +=1


    plt.figure(figsize=(9, 6))
    plt.bar(x,height=u,color = "xkcd:sky blue")
    for a,b in zip(x, u):
        if(a%2==0):
            text = str(alp[a]).upper()+": "+str(int(u[a]))
            plt.text(a, b, text,fontweight='bold')
    plt.xticks(x, alp)
    plt.ylabel('Nums')
    plt.title('Character Statistics of {}'.format(all_files[i]))
    plt.show()

# overall 
plt.figure(figsize=(11, 6))
plt.bar(x,height=u_all,color='violet')
for a,b in zip(x, u_all):
    if(a%2==0):
        text = str(alp[a]).upper()+": "+str(int(u_all[a]))
        plt.text(a, b, text,fontweight='bold')
plt.xticks(x, alp)
plt.ylabel('Nums')
plt.title('Character Statistics of the complete work')
plt.show()

for ele in range(0,26):
    print('Complete works --- number of {}: {}' .format(alp[ele], int(u_all[ele])))
    
print("Running time %s seconds" % (time.time() - start_time))

