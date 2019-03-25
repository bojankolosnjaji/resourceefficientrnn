############################################################################################
#Tensorflow RNN with attention model, for windows data
#Arguments: num_epochs learning_rate beta_1 beta_2 learning_rate_cost

############################################################################################


import numpy as np
import sklearn
import tensorflow as tf
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

num_epochs = int(sys.argv[1]) #50 # we can test

epsilon_a_t = 0.1

cost_steps=100
batch_size = 10
batch_size_per_class = 10
num_classes = 2
learning_rate = float(sys.argv[2]) #0.01 # 
epsilon = 10**-8
beta_1 = float(sys.argv[3]) # 0.9
beta_2 = float(sys.argv[4]) # 0.999
lambda_a = float(sys.argv[5]) # 0.0001 
learning_rate_cost = lambda_a
c=1
n_data = len(data_in)                #n     
n_seq = data_in.shape[1]             #t: time steps                                                                         
n_syscall = data_in.shape[2]         #h: system call and args per time step

anomaly_threshold = 10

print "Number of data points: {0}".format(n_data)
print "Sequence size:{0}".format(n_seq)
print "Number of syscalls:{0}".format(n_syscall)

## model params                                                                                                                                                                       
  
initializer = tf.random_normal_initializer(0.5,0.1)
initializer_ones = tf.random_uniform_initializer(0, 1)
batch_x = tf.placeholder(name="batch_x", shape=(batch_size, n_seq, n_syscall), dtype=tf.float32)
batch_x_noisy = tf.placeholder(name="batch_x_noisy", shape=(batch_size, n_seq, n_syscall), dtype=tf.float32)
h0 = tf.placeholder(name="h0", shape=(batch_size, n_syscall), dtype=tf.float32)
a_init = tf.constant(1.0, dtype=tf.float32, shape=(batch_size, n_syscall), name="a_init")

b_h = tf.get_variable("b_h", shape=[n_syscall], initializer=initializer, dtype=tf.float32)
b_t = tf.get_variable("b_t", shape=[n_syscall], initializer=initializer, dtype=tf.float32)

W_hh = tf.get_variable("W_hh", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ih = tf.get_variable("W_ih", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ho = tf.get_variable("W_ho", shape=[n_syscall, num_classes], initializer=initializer, dtype=tf.float32)
b_o  = tf.get_variable("b_o",  shape=[num_classes], initializer=initializer, dtype=tf.float32)

W_a = tf.get_variable("W_a", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_xa = tf.get_variable("W_xa", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_aa = tf.get_variable("W_aa", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)

W_u = tf.get_variable("W_u", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)

## model                               

num_layers=3                                                                                                                          
  
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
num_layers =1

cell_list = []
for i in range(num_layers):
  cell_list.append(tf.nn.rnn_cell.LSTMCell(n_syscall))

lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)

current_State = rnn_tuple_state
outputs_ha = []
a_t_sigmoid = a_init
attention_sigmoid_list = []

for i in range(n_seq):
  x_t = batch_x_noisy[:,i,:] #nxh
  
  a_t = tf.matmul(h_prev, W_a) + tf.matmul(x_t,W_xa) + tf.matmul(a_t_sigmoid,W_aa) +b_t #hx1
  
  attention_vector.append(a_t)
  attention_sigmoid_list.append(a_t_sigmoid)
  
  g_t = tf.multiply(x_t, a_t_sigmoid) # Hadamard Product (only when training)                                                                                                   

  a_t_sigmoid = tf.nn.relu(tf.nn.sigmoid(a_t)-epsilon_a_t)
  
  output, state = lstm_cell(g_t, state)
  outputs_ha.append(output)
  layers_ha.append(state[-1][-1])
 

a_t_stack = tf.stack(attention_sigmoid_list, axis=2)

attention_vector_sum=tf.add_n(attention_vector)                                                                                              
 
f_cost =tf.reduce_mean(tf.norm(tf.multiply(tf.tile(tf.expand_dims(feature_cost,1), [1, 10, 1]),tf.stack(attention_vector,0)), ord=1, axis=1))

losses = tf.losses.mean_squared_error(batch_x, tf.transpose(tf.stack(outputs_ha), perm=[1,0,2]))

errors = tf.norm(batch_x - tf.transpose(tf.stack(outputs_ha), perm=[1,0,2]), axis=[1,2])

mean_losses = tf.reduce_mean(losses)

total_loss = mean_losses + f_cost
                                                                                      
  
grads = tf.gradients(total_loss, W_hh)[0]          
       
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_losses)
train_step_cost =tf.train.GradientDescentOptimizer(learning_rate_cost).minimize(f_cost) 

cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
loss_list = []
test_results = []
a_t_res_list = []
a_t_list = []                                                                                                                                                                         

for iter_num, (train_index, test_index) in enumerate(skf.split(data_in, labels_in)):
    print "Test: {0}".format(iter_num)
    sess = tf.Session(config=cfg)
    sess.run(tf.global_variables_initializer())

    X_train, X_test = data_in[train_index], data_in[test_index]
    y_train, y_test = labels_in[train_index], labels_in[test_index]

    indices_train_malicious = (np.where(y_train==1)[0])
    indices_train_benign = (np.where(y_train==0)[0])

    print "Number of benign:{0}".format(len(indices_train_benign))
    print "Number of malicious:{0}".format(len(indices_train_malicious))

    num_benign = len(indices_train_benign)
    record_shape = np.shape(X_train[0,:,:])

    X_train_noisy = np.copy(X_train)
    
    for i in range(num_benign):
     
        X_train_noisy[indices_train_benign[i],:,:] = X_train_noisy[indices_train_benign[i],:,:] + np.random.normal(0,0.0001, record_shape) # adding noise
        X_train_noisy[X_train_noisy<0] = 0
    
    att_sum_all_old=0
    for epoch in range(num_epochs):
        draw_int_malicious=np.random.permutation(indices_train_malicious)[0:batch_size_per_class] # draw a malicious batch
        draw_int_benign = np.random.permutation(indices_train_benign)[0:batch_size_per_class] # draw a benign batch
        
        X_train_batch_noisy = X_train_noisy[draw_int_benign,:,:]
        X_train_batch = X_train[draw_int_benign,:,:]

        h_init = np.zeros((batch_size, n_syscall), dtype=np.float32)
        acc = 0
        
        for cost_step in range(cost_steps):
          sess.run(train_step_cost, feed_dict={batch_x:X_train_batch,batch_x_noisy:X_train_batch_noisy, h0:h_init}) 

        sess.run(train_step, feed_dict={batch_x:X_train_batch,batch_x_noisy:X_train_batch_noisy, h0:h_init})
        total_loss_val =  total_loss.eval(session=sess, feed_dict={batch_x:X_train_batch, batch_x_noisy:X_train_batch_noisy, h0:h_init})
        f_cost_val= f_cost.eval(session=sess, feed_dict={batch_x:X_train_batch, batch_x_noisy:X_train_batch_noisy, h0:h_init})
        mean_loss_val = mean_losses.eval(session=sess, feed_dict={batch_x:X_train_batch, batch_x_noisy:X_train_batch_noisy, h0:h_init})

        print("Epoch: {:5d}, Loss: {:.10f}, Cost: {:.10f} Mean CE: {:.10f}".format(epoch, total_loss_val, f_cost_val, mean_loss_val))
        loss_list.append(total_loss_val)

    ## validation                                                                      
        
    v_loss_list = []
    v_probs_x_list = []
    v_acc_list = []
    v_labels_list = []
    v_error_list = []
    v_true_list = []
    num_test = len(y_test)
    y_test_d = np.zeros((num_test,2))
    for i in range(num_test):
      y_test_d[i,int(y_test[i])]=1 # 0-benign, 1-malicious
    num_batches = int(np.floor(num_test/float(batch_size)))
    print "Number of test batches:{0}".format(num_batches)
    a_t_stack_list = []
    for batch_id in range(num_batches):
      
      a_t_stacks, v_loss, v_errors = sess.run([a_t_stack, total_loss, errors], feed_dict={batch_x:X_test[batch_id*batch_size:(batch_id+1)*batch_size,:,:], batch_x_noisy:X_test[batch_id*batch_size:(batch_id+1)*batch_size,:,:], h0:h_init})
      v_loss_list.append(v_loss)
      v_error_list.append(v_errors)
      v_true_list.append(y_test_d[batch_id*batch_size:(batch_id+1)*batch_size,:])
      a_t_stack_list.append(a_t_stacks)                                                                                                                                                

    if (num_batches*batch_size<len(y_test)):
      a_t_stacks, v_loss, v_errors = sess.run([a_t_stack, total_loss, errors], feed_dict={batch_x:X_test[num_test-batch_size:num_test,:,:], batch_x_noisy:X_test[batch_id*batch_size:(batch_id+1)*batch_size,:,:], h0:h_init})
      rest_size = len(y_test)-num_batches*batch_size
      v_loss_list.append(v_loss)
      v_error_list.append(v_errors)
      v_true_list.append(y_test_d[num_test-batch_size:num_test,:])
      a_t_stack_list.append(a_t_stacks)
   
    v_loss = np.sum(v_loss_list)
    v_error_all = np.concatenate(v_error_list)
    v_true_all = np.concatenate(v_true_list)

    confusion_matrix = np.zeros((2,2))

    for i in range(len(v_true_all)):
      if (v_error_all[i])>anomaly_threshold:
        predicted_class=1
      else:
        predicted_class=0

      actual_class = v_true_all[i,1]

      confusion_matrix[int(predicted_class), int(actual_class)]+=1
    

    print v_error_list
    print v_true_list

    print "Loss:{0}".format(v_loss)
    num_tests = len(v_error_all)
    for i in range(num_tests):
      print "{0} {1}".format(v_error_all[i], v_true_all[i,:])


    tp = confusion_matrix[1,1]
    tn = confusion_matrix[0,0]
    fp = confusion_matrix[1,0]
    fn = confusion_matrix[0,1]

    precision = tp/float(tp+fp)

    recall = tp/float(tp+fn)
    print "Precision:{0}, Recall:{0}, F1:{0}".format(precision,recall,2*tp/(2*tp+fp+fn))
    
    print confusion_matrix

    test_results.append(confusion_matrix)

    a_t_res_list.append([len(np.nonzero(a_t_stacks)[0]), np.size(a_t_stacks)])
    a_t_list.append(a_t_stacks)                                                                                                                                                         


np.savez('./results/ae/results_ae_{0}_{1}.npz'.format(lambda_a, learning_rate), test_results)
np.savez('./results/ae/a_t_ae_{0}_{1}.npz'.format(lambda_a, learning_rate), np.asarray(a_t_res_list))
np.savez('./results/ae/a_t_raw_ae_{0}_{1}.npz'.format(lambda_a, learning_rate), np.asarray(a_t_list))                                                                                         

    
    

