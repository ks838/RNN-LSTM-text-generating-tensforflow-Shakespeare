# RNN-LSTM-text-generating-using-tensorflow-Shakespeare

@ Author:  Kai Song, ks838@cam.ac.uk

@ Notes:   What does this small project do?
            1. In '1-basic-statistics','2-statistics-of-characters' and '3-Statistics-of-Words', I did 
               basic character and word statistics of Shakespeare's complete works, such that I could 
               believe these works are writtern by one great Shakepeare.
            1. I used Recurrent Neural Network-LSTM to do text generating. I wrote the LSTM core part in 
               a relatively transparent way according Reference [1], indstead of using more 
               abstract/advanced tensorfow functions.
            2. The results in 'output.txt' were generated using the first 10 texts (~ 250,000 words) of 
               the complete works. You could use one comedy or any text for primacy testing, without torturing 
               your laptop too much.
            3. I used one-hot representation for characters, instead of words and phrases because the former 
               is much simpler (smaller vectors). Specially, for Shakespeare's complete works, the one-hot vectors
               for char is just 56 \times 1, while the word-based vectors would ~30,000 (How impressive!). But I think the letter one is more like human's thinking and writing. Shakespeare is really great just in this respect:-) 

            4. I used python     3.5.3, 
                      tensorflow 1.3.0

@ Refs:
            1. For LSTM, please refer to the famous paper "Recurrent Neural Network Regularization" by 
               W Zaremba et al.
            2. Why using sigmoid and tanh as the activation functions in LSTM?
               I found a explaination on https://www.quora.com/
            3. My tf codes are based on 
               https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/3_NeuralNetworks
            4. The complete works were downloaded at 
               https://github.com/martin-gorner/tensorflow-rnn-shakespeare

@ Reconmanded blogs:
            1. https://www.youtube.com/watch?v=9zhrxE5PQgY
               There, Siraj used only numpy, giving a rather nice lecture on LSTM. 
            2. On LSTM parameter tuning: https://deeplearning4j.org/lstm.html
