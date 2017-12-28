'''
@ Author:  Kai Song, ks838@cam.ac.uk

@ Notes:   What does this small project do?
           Word statistics for the complete works of Shakespeare.
'''

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab

import re
from collections import Counter

from os import listdir
from os.path import isfile, join
import codecs

print(__doc__)
path_shake = '../complete_works/'
all_files = [f.replace('.txt','') for f in listdir(path_shake) if isfile(join(path_shake, f))]
n_files = len(all_files)
print("Total word numbers of Shakepeare's complete works: ", n_files)
print('\n')

# words for the complete work
words = []
N = 10 # top N
for i in range(n_files):
    file_i_path = '../complete_works/'+ all_files[i]+'.txt'
    file_i = codecs.open(file_i_path, "r",encoding='utf-8', errors='ignore')
    text_i = file_i.read().lower()
    word_i = re.findall(r'\w+', text_i)
    words +=word_i
    word_i_count = len(word_i)
    vocab_i_count = len(set(word_i))
    top_N = Counter(words).most_common(N)
    print('-'*30 + str(i+1) + '-'*30)
    print(top_N)
    print('No. of words in %s is %d'%(all_files[i],word_i_count))
    print('No. of different words in %s is %d'%(all_files[i],vocab_i_count))

    top_word = []
    Nums= []
    for ele in top_N:
        top_word.append(ele[0])
        Nums.append(ele[1])
    
    # plot
    fig, ax = plt.subplots(figsize=(11, 7))  
    index = np.arange(N)
    width = 0.5
    ax.barh(index, Nums,width,color="blue")
    for j, v in enumerate(Nums):
        ax.text(v + 50, j + .07, str(v), fontweight='bold')
        ax.text(v + 50, j - .22, str('%.4f'%(v/word_count*100))+'%', fontweight='bold')
    ax.set_yticks(index)#+width/2)
    ax.set_yticklabels(top_word,fontweight='bold')
    plt.ylabel('Top %d Words'%N,fontweight='bold')
    plt.title('Word Statistics of %s: No. of words=%d'%(all_files[i],word_i_count))
    plt.show()
    
words_count = len(words)
vocabs_count = len(set(words))  
print('\nThe total number of words Shakespeare used in the complete work: ',words_count)
print('\nThe total number of different words Shakespeare used in the complete work: ',vocabs_count)