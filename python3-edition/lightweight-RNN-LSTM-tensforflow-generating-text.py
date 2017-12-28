'''
@ Author:  Kai Song, ks838@cam.ac.uk

@ Notes:   What does this small project do?
           1. I used Recurrent Neural Network-LSTM to do text generating. I wrote the LSTM core part in 
              a relatively transparent way according Reference [1], indstead of using more 
              abstract/advanced tensorfow functions.
           2. The results in 'output.txt' were generated using the first 10 texts (~ 250,000 words) of 
              the complete works. You could use one comedy or any text for primacy testing, without torturing 
              your laptop too much.

@ Refs:
           1. For LSTM, please refer to the famous paper "Recurrent Neural Network Regularization" by 
              W Zaremba et al.
           2. Why using sigmoid and tanh as the activation functions in LSTM?
              I found a explaination on https://www.quora.com/
           3. https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/3_NeuralNetworks

@ Reconmanded blogs:
           1. https://www.youtube.com/watch?v=9zhrxE5PQgY
              There, Siraj used only numpy, giving a rather nice lecture on LSTM. 
           2. On LSTM parameter tuning: https://deeplearning4j.org/lstm.html

'''
import numpy as np
import tensorflow as tf
import sys
import codecs
from os import listdir
from os.path import isfile, join

print(__doc__)
path_shake = '../complete_works/'
all_files = [f.replace('.txt','') for f in listdir(path_shake) if isfile(join(path_shake, f))]
n_files = len(all_files)
print("n_files = ",n_files)
raw_text = []
for i in range(10):
    #raw_text = open('../sss.txt').read().lower()
    file_name = '../complete_works/'+ all_files[i]+'.txt'
    text_i = codecs.open(file_name, "r",encoding='utf-8', errors='ignore').read().lower()
    raw_text +=text_i
#raw_text = open('/Users/stusk/machine_learning/sk-projects/shakespeare-statitics/sss.txt').read().lower()
print('The number of characters in our raw text:', len(raw_text))

#print('head of text:')
#print(raw_text[:50])
#assert(1>2)
chars = sorted(list(set(raw_text)))
char_size = len(chars)
print('number of different characters:', char_size)
print(chars)

char_to_ix = dict((c, i) for i, c in enumerate(chars))
ix_to_char = dict((i, c) for i, c in enumerate(chars))


seq_length = 50

data_in = []
data_out = []
for i in range(0, len(raw_text) - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    #out: just the next char of seq_in
    seq_out = raw_text[i + seq_length]
    data_in.append(seq_in)
    data_out.append(seq_out)


X = np.zeros((len(data_in), seq_length, char_size))
y = np.zeros((len(data_in), char_size))
for i, sect_i in enumerate(data_in):
    for j, char_j in enumerate(sect_i):
        X[i, j, char_to_ix[char_j]] = 1
    y[i, char_to_ix[data_out[i]]] = 1


# Training Parameters
learning_rate = 0.01
batch_size = 212
nsteps = 40000
hidden_nodes = 154


print('training data size:', len(X))
print('No. of epoches: %.2f'%(nsteps/len(X)))
print('No. of batches per epoch:', int(len(X)/batch_size))

'''
tf.graph here is unnecessary since we have only one, 
but it's a good practice to follow.
If we start to work with many graphs, 
it's easier to understand where ops and vars are placed
'''
graph = tf.Graph()
with graph.as_default():
    # the weights and biases
    W = {
        #Input gate: weights for input, and input from previous output
        'ii': tf.Variable(tf.random_normal([char_size, hidden_nodes])),
        'io': tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes])),
        #Forget gate: weights for input, previous output
        'fi': tf.Variable(tf.random_normal([char_size, hidden_nodes])),
        'fo': tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes])),
        #Output gate: weights for input, previous output
        'oi': tf.Variable(tf.random_normal([char_size, hidden_nodes])),
        'oo': tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes])),
        #Memory cell: weights for input, previous output
        'ci': tf.Variable(tf.random_normal([char_size, hidden_nodes])),
        'co': tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes])),
        # output
        'out': tf.Variable(tf.random_normal([hidden_nodes, char_size],mean=-0.1,stddev=0.1))
    }
    biases = {
        'i': tf.Variable(tf.zeros([1, hidden_nodes])),
        'f': tf.Variable(tf.zeros([1, hidden_nodes])),
        'o': tf.Variable(tf.zeros([1, hidden_nodes])),
        'c': tf.Variable(tf.zeros([1, hidden_nodes])),
        'out': tf.Variable(tf.zeros([char_size]))
    }
    # LCTM Cell
    # iteration: h^{l−1}_t,h^l_{t-1} ,c^l_{t−1} -> h^l_t,c^l_t
    def RNN_LSTM(h_state_0, h_state_1, cell):
        # Sigmoid is usually used as the gating function for the 3 gates(in,  out,  forget)  in LSTM.
        # Dealing with vanishing gradient problem for lstm is different than that for a feed forward deep net.  
        # Here,  it's resolved by the structure of the lstm network,  
        # specifically the various gates and a memory cell.
        input_gate  = tf.sigmoid(tf.matmul(h_state_0, W['ii']) + tf.matmul(h_state_1, W['io']) + biases['i'])
        forget_gate = tf.sigmoid(tf.matmul(h_state_0, W['fi']) + tf.matmul(h_state_1, W['fo']) + biases['f'])
        output_gate = tf.sigmoid(tf.matmul(h_state_0, W['oi']) + tf.matmul(h_state_1, W['oo']) + biases['o'])
        modulation_gate= tf.tanh(tf.matmul(h_state_0, W['ci']) + tf.matmul(h_state_1, W['co']) + biases['c'])
        cell = forget_gate * cell + input_gate * modulation_gate
        h_state_out = output_gate * tf.tanh(cell)
        return h_state_out, cell

    h_state_0 = tf.zeros([batch_size, seq_length, char_size])
    labels = tf.placeholder("float", [batch_size, char_size])

    def logits_and_loss():
        h_state_1 = tf.zeros([batch_size, hidden_nodes])
        cell = tf.zeros([batch_size, hidden_nodes])    

        for i in range(seq_length):
            h_state_1, cell = RNN_LSTM(h_state_0[:, i, :], h_state_1, cell)
            # We concatenate them together to calculate the logits and loss
            if i == 0:
                h_state_1_i = h_state_1
                h_state_0_i = h_state_0[:, i+1, :]
            elif (i != seq_length - 1):
                h_state_1_i = tf.concat([h_state_1_i, h_state_1],0)
                h_state_0_i = tf.concat([h_state_0_i, h_state_0[:, i+1, :]],0)
            else:
                h_state_1_i = tf.concat([h_state_1_i, h_state_1],0)
                h_state_0_i = tf.concat([h_state_0_i, labels],0)
            
        logits = tf.matmul(h_state_1_i, W['out']) + biases['out']
        loss   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          logits=logits, 
                          labels=h_state_0_i))
        return logits, loss
    
    #Optimizer
    logits,loss = logits_and_loss()
    optimizer0 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer  = optimizer0.minimize(loss)

    # for the on-the-fly Test
    test_h_state_0 = tf.Variable(tf.zeros([1, char_size]))
    test_h_state_1 = tf.Variable(tf.zeros([1, hidden_nodes]))
    test_cell      = tf.Variable(tf.zeros([1, hidden_nodes]))
    
    #re-initialize at the beginning of each test
    reset_test_cell = tf.group(test_h_state_1.assign(tf.zeros([1, hidden_nodes])), 
                                test_cell.assign(tf.zeros([1, hidden_nodes])))

    #RNN LSTM
    test_h_state_1, test_cell = RNN_LSTM(test_h_state_0, test_h_state_1, test_cell)
    test_prediction = tf.nn.softmax(tf.matmul(test_h_state_1, W['out']) + biases['out'])

#Create a checkpoint directory
   #True if the path exists, whether its a file or a directory.
checkpoint_file = 'checkpoint_file'
if tf.gfile.Exists(checkpoint_file):
    tf.gfile.DeleteRecursively(checkpoint_file)
tf.gfile.MakeDirs(checkpoint_file)

# the seed for the on-the-fly testing
test_seed = 'The first principle is that you must not fool yourself.'.lower()
fout1 = open('output.dat','w')

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    shift = 0
    saver = tf.train.Saver()
    print('')
    print('test_seed: ',test_seed)

    for step in range(nsteps):
        shift = shift % len(X)
        if shift <= (len(X) - batch_size):
            batch_h_state_0 = X[shift: shift + batch_size]
            batch_labels = y[shift: shift + batch_size]
            shift += batch_size
        else:#the final batch in an epoch
            complement = batch_size - (len(X) - shift)
            batch_h_state_0 = np.concatenate((X[shift: len(X)], X[0: complement]))
            batch_labels = np.concatenate((y[shift: len(X)], y[0: complement]))
            shift = np.random.choice(batch_size)# start the next epoch with a random start char
        _, training_loss = sess.run([optimizer, loss], feed_dict={h_state_0: batch_h_state_0, labels: batch_labels})
        
        if step % 200 == 0:
            print('\n'+'-' * 15 +'training loss at step %d: %.2f' % (step, training_loss)+'-' * 15)
            fout1.write('\n'+'-' * 15 +'training loss at step %d: %.2f' % (step, training_loss)+'-' * 15+'\n')
            reset_test_cell.run()
            test_generated = ''
            
            for i in range(len(test_seed) - 1):
                test_X = np.zeros((1, char_size))
                # each char in our test_seed is a vector(one-hot)
                test_X[0, char_to_ix[test_seed[i]]] = 1.0
                sess.run(test_prediction, feed_dict={test_h_state_0: test_X})

            test_X = np.zeros((1, char_size))
            # use the last char of the seed as a start of our on-the-fly prediction
            test_X[0, char_to_ix[test_seed[-1]]] = 1.0
            stdout1 = []
            for i in range(200):
                prob_distribution = test_prediction.eval({test_h_state_0: test_X})[0]
                next_char_one_hot = np.zeros((char_size))
                #pick one with the higher probability 
                ix = np.random.choice(range(char_size), p=prob_distribution.ravel())
                next_char_one_hot[ix] = 1.0
                next_char = ix_to_char[ix]
                # if you want to output the results to a file,use
                # python the_present.py > filename
                sys.stdout.write(next_char)
                fout1.write(next_char)
                test_X = next_char_one_hot.reshape((1,char_size))
                

            saver.save(sess, checkpoint_file + '/model', global_step=step)
    fout1.close()
    print('\nThe weights of our RNN-LSTM have been saved in ',checkpoint_file) 
    print('\nDone Successfully!')       