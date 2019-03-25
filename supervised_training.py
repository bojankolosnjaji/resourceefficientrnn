############################################################################################
#Tensorflow RNN with attention model and sparsity, for windows data
#Arguments: num_epochs learning_rate beta_1, beta_2, lambda_a 

############################################################################################


import numpy as np
import sklearn
import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import sys

np.random.seed(1337)

tf.set_random_seed(1337)

data_in = np.load('data_new.npy')
labels_in = np.load('labels_new.npy')

skf = StratifiedKFold(n_splits=3, shuffle=True)

print "Model construction..."

epsilon_a_t = 0.1

num_epochs = int(sys.argv[1]) #50 # we can test
cost_steps=100
batch_size = 10
batch_size_per_class = 5
num_classes = 2
learning_rate = float(sys.argv[2]) #0.01 # we can test
epsilon = 10**-8
beta_1 = float(sys.argv[3]) # 0.9
beta_2 = float(sys.argv[4]) # 0.999
lambda_a = float(sys.argv[5]) # 0.0001  # we can test
c=1
n_data = len(data_in)                #n     
n_seq = data_in.shape[1]             #t: time steps                                                                         
n_syscall = data_in.shape[2]         #h: system call and args per time step  

print "Number of data points: {0}".format(n_data)
print "Sequence size:{0}".format(n_seq)
print "Number of syscalls:{0}".format(n_syscall)

## model params                                                                                                                                                                       
  
initializer = tf.random_uniform_initializer(0,0)
initializer_ones = tf.random_uniform_initializer(0, 1)
batch_x = tf.placeholder(name="batch_x", shape=(batch_size, n_seq, n_syscall), dtype=tf.float32)
batch_y = tf.placeholder(name="batch_y", shape=(batch_size, num_classes), dtype=tf.int32)
h0 = tf.placeholder(name="h0", shape=(batch_size, n_syscall), dtype=tf.float32)
a_init = tf.get_variable("a_init", shape=(batch_size, n_syscall), dtype=tf.float32, initializer=tf.keras.initializers.Ones)
b_h = tf.get_variable("b_h", shape=[n_syscall], initializer=initializer, dtype=tf.float32)
b_t = tf.get_variable("b_t", shape=[n_syscall], initializer=initializer, dtype=tf.float32)

W_hh = tf.get_variable("W_hh", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ih = tf.get_variable("W_ih", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ho = tf.get_variable("W_ho", shape=[n_syscall, num_classes], initializer=initializer, dtype=tf.float32)
b_o  = tf.get_variable("b_o",  shape=[num_classes], initializer=initializer, dtype=tf.float32)

W_a = tf.get_variable("W_a", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_xa = tf.get_variable("W_xa", shape=[n_syscall, n_syscall], initializer=initializer)
W_aa = tf.get_variable("W_aa", shape=[n_syscall, n_syscall], initializer=initializer)

W_u = tf.get_variable("W_u", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)

## model                               

num_layers=1                                                                                                                           
layers_ha= []
attention_vector = []
feature_cost = tf.constant(1, dtype=np.float32, shape=[n_seq, n_syscall])
h_prev = h0
hidden_state = tf.zeros([batch_size, n_syscall])
current_state = tf.zeros([batch_size, n_syscall])
state = hidden_state, current_state

init_state = tf.zeros([num_layers, 2, batch_size, n_syscall], dtype=tf.float32)

state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

state = rnn_tuple_state
                                                               
###attention layer                            

lstm_list = []
for i in range(num_layers):
  lstm_list.append(tf.nn.rnn_cell.LSTMCell(n_syscall))

lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_list, state_is_tuple=True)

current_State = rnn_tuple_state
       
a_t_sigmoid = a_init
attention_sigmoid_list = []

for i in range(n_seq):
  x_t = batch_x[:,i,:] #nxh
  a_t = tf.matmul(h_prev, W_a) + tf.matmul(x_t,W_xa) + tf.matmul(a_t_sigmoid,W_aa) + b_h #hx1
  
  attention_vector.append(a_t_sigmoid)
  attention_sigmoid_list.append(a_t_sigmoid)
  g_t = tf.multiply(x_t, a_t_sigmoid) # Hadamard Product (only when training)
  a_t_sigmoid = tf.nn.relu(tf.nn.sigmoid(a_t)-epsilon_a_t)



  output, state = lstm_cell(g_t, state)
  layers_ha.append(state[-1][-1])


dropout_rate = tf.placeholder_with_default(1.0, shape=())

a_t_stack = tf.stack(attention_sigmoid_list, axis=2)

### maxpooling                                                                                                                                                                       
hs = tf.transpose(tf.convert_to_tensor(layers_ha, dtype=tf.float32))

h_max = tf.nn.avg_pool(tf.reshape(hs, [n_syscall, batch_size, n_seq, 1]), [1, 1, n_seq, 1], [1, 1, n_seq, 1], "VALID")
h_max = tf.transpose(tf.reshape(h_max, [n_syscall, batch_size]))
### loss calculation                               
                                     
h_max_dropout =  tf.nn.dropout(h_max, dropout_rate)                                                            
attention_vector_sum=tf.add_n(attention_vector)                                                                                              
f_cost = tf.reduce_mean(tf.norm(tf.multiply(tf.tile(tf.expand_dims(feature_cost,1), [1, 10, 1]),tf.stack(attention_vector,0)), ord=1, axis=1))
                                                   
### loss calculation                                                                                                                                                                  
logits_series = tf.matmul(h_max_dropout, W_ho) + b_o #+ f_cost + epsilon

class_weights=tf.constant([[1.0], [1.0]], dtype=tf.float32, name='ClassWeights')
sample_weights = tf.squeeze(tf.matmul(tf.cast(batch_y, dtype=tf.float32), class_weights))
losses = tf.losses.softmax_cross_entropy(onehot_labels=batch_y, logits=logits_series, weights=sample_weights)

                                                  
mean_losses = tf.reduce_mean(losses)

total_loss = mean_losses + f_cost
### accuracy calculation                                                                                                
probs_x = tf.nn.softmax(logits_series)
labels_x = tf.cast(tf.argmax(probs_x, 1), tf.float32)
batch_y_binary = tf.cast(tf.argmax(batch_y, 1), tf.float32)
compare = tf.cast(tf.equal(tf.cast(batch_y_binary, tf.float32), labels_x), tf.float32)
accuracy = tf.div(tf.reduce_sum(compare), batch_size, "Accuracy")
### training                                                                           
                                                                                     
  
grads = tf.gradients(total_loss, W_hh)[0]

learning_rate_cost = lambda_a
       
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_losses)
train_step_cost = tf.train.GradientDescentOptimizer(learning_rate_cost).minimize(f_cost)

cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
loss_list = []

test_results = []
a_t_res_list=[]
a_t_list = []

for iter_num, (train_index, test_index) in enumerate(skf.split(data_in, labels_in)):
    print "Test: {0}".format(iter_num)
    sess = tf.Session(config=cfg)
    sess.run(tf.global_variables_initializer())

    X_train, X_test = data_in[train_index], data_in[test_index]
    y_train, y_test = labels_in[train_index], labels_in[test_index]


    indices_train_malicious = (np.where(y_train==1)[0])
    indices_train_benign = (np.where(y_train==0)[0])
    att_sum_all_old=0
    for epoch in range(num_epochs):
        draw_int_malicious=np.random.permutation(indices_train_malicious)[0:batch_size_per_class] # draw 10 for a batch
        draw_int_benign = np.random.permutation(indices_train_benign)[0:batch_size_per_class] # draw 10 for a batch
        
        X_train_batch_malicious = X_train[draw_int_malicious,:,:]
        X_train_batch_benign = X_train[draw_int_benign,:,:]
        y_train_batch_malicious = np.ones(len(draw_int_malicious))
        y_train_batch_benign = np.zeros(len(draw_int_benign))
        X_train_batch = np.concatenate((X_train_batch_malicious, X_train_batch_benign), axis=0)
        y_train_batch = np.concatenate((y_train_batch_malicious, y_train_batch_benign), axis=0)

        batch_size_y = len(y_train_batch)
        y_train_batch_d = np.zeros((batch_size_y, 2))
        for i in range(batch_size_y):
          y_train_batch_d[i, int(y_train_batch[i])]=1


        h_init = np.zeros((batch_size, n_syscall), dtype=np.float32)
        acc = 0
        
        for cost_step in range(cost_steps):
          sess.run(train_step_cost, feed_dict={batch_x:X_train_batch,batch_y:y_train_batch_d, h0:h_init, dropout_rate:1.0})

        sess.run(train_step, feed_dict={batch_x:X_train_batch,batch_y:y_train_batch_d, h0:h_init, dropout_rate:1.0})

        total_loss_val =  total_loss.eval(session=sess, feed_dict={batch_x:X_train_batch, batch_y:y_train_batch_d, h0:h_init})
        f_cost_val= f_cost.eval(session=sess, feed_dict={batch_x:X_train_batch, batch_y:y_train_batch_d, h0:h_init})

        mean_loss_val = mean_losses.eval(session=sess, feed_dict={batch_x:X_train_batch, batch_y:y_train_batch_d, h0:h_init})
        att_vec_list = []

        print("Epoch: {:5d}, Loss: {:.10f}, Cost: {:.10f} Mean CE: {:.10f}".format(epoch, total_loss_val, f_cost_val, mean_loss_val))

        loss_list.append(total_loss_val)

    ## validation                                                                      
        
    v_loss_list = []
    v_probs_x_list = []
    v_acc_list = []
    v_labels_list = []
    num_test = len(y_test)
    y_test_d = np.zeros((num_test,2))
    for i in range(num_test):
      y_test_d[i,int(y_test[i])]=1
    num_batches = int(np.floor(num_test/float(batch_size)))
    print "Number of test batches:{0}".format(num_batches)
    a_t_stack_list = []
    for batch_id in range(num_batches):
      a_t_stacks, v_loss, v_probs_x, v_labels_x, v_acc = sess.run([a_t_stack, total_loss, probs_x, labels_x, accuracy], feed_dict={batch_x:X_test[batch_id*batch_size:(batch_id+1)*batch_size,:,:], batch_y:y_test_d[batch_id*batch_size:(batch_id+1)*batch_size,:], h0:h_init})
      v_loss_list.append(v_loss)
      v_probs_x_list.append(v_probs_x)
      v_acc_list.append(v_acc)
      v_labels_list.append(v_labels_x)
      a_t_stack_list.append(a_t_stacks)
    if (num_batches*batch_size<len(y_test)):
      a_t_stacks, v_loss, v_probs_x, v_labels_x, v_acc = sess.run([a_t_stack, total_loss, probs_x, labels_x, accuracy], feed_dict={batch_x:X_test[num_test-batch_size:num_test,:,:], batch_y:y_test_d[num_test-batch_size:num_test,:], h0:h_init})
      rest_size = len(y_test)-num_batches*batch_size
      v_loss_list.append(v_loss)
      v_probs_x_list.append(v_probs_x[batch_size-rest_size:,:])
      v_acc_list.append(v_acc)
      v_labels_list.append(v_labels_x[batch_size-rest_size:])
      a_t_stack_list.append(a_t_stacks)

    
    v_loss = np.sum(v_loss_list)
    v_probs_x = np.vstack(v_probs_x_list)
    v_labels_x = np.concatenate(v_labels_list)
    v_acc= np.mean(v_acc_list)
    a_t_stacks = np.stack(a_t_stack_list)
    precision, recall, _, _ = precision_recall_fscore_support(y_test.flatten(), v_labels_x, average="binary", pos_label=1)
    fpr, tpr, thresholds = roc_curve(y_test.flatten(), v_probs_x[:,1], pos_label=1)

    confusion_matrix_test = confusion_matrix(y_test.flatten(), v_labels_x)

    test_results.append(confusion_matrix_test)

    v_auc = auc(fpr, tpr)
    print "Sum a_t:{0}".format(np.sum(a_t_stacks, axis=(0,1)))
    print "Nonzero a_t:{0}/{1}".format(len(np.nonzero(a_t_stacks)[0]), np.size(a_t_stacks))
    a_t_res_list.append([len(np.nonzero(a_t_stacks)[0]), np.size(a_t_stacks)])
    a_t_list.append(a_t_stacks)
    print("Validation>> Loss:{:.2f}, Accuracy: {:.2f}%, AUC: {:.2f}, Precision:{:.2f}%, Recall:{:.2f}%".format(v_loss, v_acc*100, v_auc, precision*100, recall*100))
    print("Confusion matrix{0}".format(confusion_matrix_test))

np.savez('./results/results_{0}_{1}.npz'.format(lambda_a, learning_rate), test_results)
np.savez('./results/a_t_{0}_{1}.npz'.format(lambda_a, learning_rate), np.asarray(a_t_res_list))
np.savez('./results/a_t_raw_{0}_{1}.npz'.format(lambda_a, learning_rate), np.asarray(a_t_list))        
        
        
    
    

