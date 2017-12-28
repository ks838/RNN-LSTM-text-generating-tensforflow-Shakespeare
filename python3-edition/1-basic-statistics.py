'''
@ Author:  Kai Song, ks838@cam.ac.uk

@ Notes:   What does this small project do?
           1. This is just the basic statistics, very simple.
'''

from os import listdir
from os.path import isfile, join


print(__doc__)

path_shake = './complete_works/'
all_files = [f.replace('.txt','') for f in listdir(path_shake) if isfile(join(path_shake, f))]
n_files = len(all_files)
print("The total number of Shakespeare's complete work: ",n_files)
print(all_files)

tragedy = 'tragedy'
n_tragedy = 0
tragedy_files=[]

poetry = 'poetry'
n_poetry = 0
poetry_files = []
sonnet = 'sonnet'
n_sonnet = 0

history = 'hist'
n_hist = 0
hist_files = []

n_comedy = 0
comedy_files = []

for i in range(num):

    if(tragedy in all_files[i]):
        n_tragedy +=1
        tragedy_files.append(all_files[i])
    elif(history in all_files[i]):
        n_hist +=1
        hist_files.append(all_files[i])
    elif(poetry in all_files[i]):
        n_poetry +=1
        poetry_files.append(all_files[i])
        if(sonnet in all_files[i]):
            n_sonnet +=1
    else:
        n_comedy +=1
        comedy_files.append(all_files[i])

print("n_tragedy = ",n_tragedy) # 12
print("The tragedies are: \n",tragedy_files)

print("n_hist = ",n_hist) # 10
print("The histories are: \n",tragedy_files)

print("n_poetry = %d,"%n_poetry, "while %d of them are sonnets."%n_sonnet) # 5
print("The poetries are: \n",poetry_files)

print("n_comedy = ",n_comedy) # 15
print("The comedies are: \n",comedy_files)

import numpy as np 
import codecs

word_count = np.zeros(n_files)
line_count = np.zeros(n_files)

for i in range(n_files):
    file_name = './complete_works/'+ all_files[i]+'.txt'
    with codecs.open(file_name, "r",encoding='utf-8', errors='ignore') as fdata:
        data = fdata.read()
        sentences_1 = data.split('\n')
        sentences_2 = [elem for elem in sentences_1 if elem !='']

        for part in sentences_2:
            word_list = part.split(' ')
            word_count[i] += len(word_list)
            line_count[i] += 1
        print(i+1)
        print("Total word numbers of "+all_files[i]+": ", int(word_count[i]))
        print("Total line numbers of "+all_files[i]+": ", int(line_count[i]))
        print('')