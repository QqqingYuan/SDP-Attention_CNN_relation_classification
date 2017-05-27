__author__ = 'PC-LiNing'

import datetime
import numpy
import tensorflow as tf
import load_data_weight
import data_helpers
from ops import conv2d
import argparse
import official_score
import math
import os


NUM_CLASSES = 19
EMBEDDING_SIZE = 200
PF_SIZE = 30
FINAL_EMBEDDING_SIZE = EMBEDDING_SIZE + PF_SIZE*2 + EMBEDDING_SIZE
NUM_CHANNELS = 1
SEED = 66478
BATCH_SIZE = 125
NUM_EPOCHS = 200
EVAL_FREQUENCY = 100
META_FREQUENCY = 100
# learning rate
learning_rate_decay = 0.5
start_learning_rate = 1e-3
decay_delta = 0.005
min_learning_rate = 5e-5
# train
steps_each_check = 500
max_document_length = 50

# test size
Test_Size = 2717
Train_Size = 8000
beta = 2.0
multiple = 1.0

# FLAGS=tf.app.flags.FLAGS
FLAGS = None


def train(argv=None):
    # load data
    print("Loading data ... ")
    pf_train, w_train = load_data_weight.load_train_data()
    pf_test, w_test = load_data_weight.load_test_data()

    x_sent = numpy.load('x_sent_Dep_50.npy')
    y_sent = numpy.load('y_sent_Dep_50.npy')

    # split to train and test .
    x_train = x_sent[:Train_Size]
    y_train = y_sent[:Train_Size]
    x_test=x_sent[Train_Size:]
    y_test=y_sent[Train_Size:]

    print(x_train.shape)
    print(x_test.shape)

    # Test record
    record_f1 = []
    record_acc = []
    record_predicts = []
    record_alpha = []

    filter_sizes = [2,3,4,5]
    filter_numbers = [300,200,100,50]

    # input
    # input is sentence
    train_data_node = tf.placeholder(tf.float32,shape=(None,max_document_length,EMBEDDING_SIZE))

    train_labels_node = tf.placeholder(tf.float32,shape=(None,NUM_CLASSES))

    train_pf_node = tf.placeholder(tf.float32,shape=(None,max_document_length,2*PF_SIZE))

    train_sdp_node = tf.placeholder(tf.float32, shape=(None, max_document_length))

    dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

    # input attention matrix
    d_c = sum(filter_numbers)
    init_att = math.sqrt(6.0 / (EMBEDDING_SIZE + d_c))
    U = tf.Variable(tf.random_uniform([EMBEDDING_SIZE, d_c],
                                                minval=-init_att,
                                                maxval=init_att,
                                                dtype=tf.float32))

    init_m = math.sqrt(6.0 / (max_document_length + NUM_CLASSES))
    M = tf.Variable(tf.random_uniform([max_document_length, NUM_CLASSES],
                                                minval=-init_m,
                                                maxval=init_m,
                                                dtype=tf.float32))

    # class embeddings matrix
    init_class = math.sqrt(6.0 / (d_c + NUM_CLASSES))
    classes_matrix = tf.Variable(tf.random_uniform([d_c, NUM_CLASSES],
                                                minval=-init_class,
                                                maxval=init_class,
                                                dtype=tf.float32))

    # model
    # data = [batch_size,n,embed]
    # pf = [batch_size,n,2*pf_dim]
    # sdp = [batch_size,n]
    def model(data,pf,sdp):
        # input attention based sdp
        # H = [batch_size x n, d_c]
        H = tf.matmul(tf.reshape(data, [-1, EMBEDDING_SIZE]), U)
        # A_P = [batch_size , n, num_class]
        A_P = tf.reshape(tf.matmul(H, classes_matrix), [-1, max_document_length, NUM_CLASSES])
        # alpha = [batch_size, 1, num_class]
        alpha = tf.expand_dims(tf.matmul(sdp, M),  axis=1)
        # alpha = [batch_size,1,n]
        alpha = tf.batch_matmul(alpha, tf.transpose(A_P, [0, 2, 1]))
        # alpha = [batch_size,n]
        alpha = tf.nn.l2_normalize(tf.squeeze(alpha), dim=-1)
        # alpha = sdp +  multiple * alpha = [batch_size,n]
        alpha_v = tf.add(sdp, tf.scalar_mul(multiple, alpha))
        # transfer to diagonal matrix
        # alpha = [batch_size,n,n]
        alpha = tf.matrix_diag(alpha_v)
        # weighted_data = [batch_size,n,embed]
        weighted_data = tf.batch_matmul(alpha, data)
        # add pf
        # pf_data = [batch_size,n,embed+2*pf_dim]
        pf_data = tf.concat(2, [data, pf])
        # combine pf_data and weighted_data
        # combine_data = [batch_size,n,(embed+2*pf_dim)+embed]
        combine_data = tf.concat(2, [pf_data, weighted_data])
        # exp_data = [batch_size,n,(embed+2*pf_dim)+embed,1]
        exp_data = tf.expand_dims(combine_data, axis=-1)
        pooled_outputs = []
        for idx, filter_size in enumerate(filter_sizes):
            conv = conv2d(exp_data,filter_numbers[idx],filter_size,FINAL_EMBEDDING_SIZE,name="kernel%d" % idx)
            # 1-max pooling,leave a tensor of shape[batch_size,1,1,num_filters]
            pool = tf.nn.max_pool(conv,ksize=[1,max_document_length-filter_size+1,1,1],strides=[1, 1, 1, 1],padding='VALID')
            pooled_outputs.append(tf.squeeze(pool))

        if len(filter_sizes) > 1:
            cnn_output = tf.concat(1,pooled_outputs)
        else:
            cnn_output = pooled_outputs[0]

        # add dropout
        # dropout_output = [batch_size,d_c]
        dropout_output = tf.nn.dropout(cnn_output,dropout_keep_prob)
        normalized_classes_matrix = tf.nn.l2_normalize(classes_matrix, dim=0)
        # scores = [batch_size,num_class]
        scores = tf.matmul(dropout_output, normalized_classes_matrix)
        return scores,  alpha_v

    # get true result and neg result
    def get_targe_neg(size,labels,predicts):
        const_v = tf.constant([NUM_CLASSES-1]*size)
        true_index = tf.cast(labels,tf.int32)
        # get non-other/other type, other is False
        other_flag = tf.not_equal(true_index, const_v)
        # split two part, non_other part and other part
        non_other = tf.slice(predicts, [0,0], [-1,NUM_CLASSES-1])
        other = tf.slice(predicts, [0,NUM_CLASSES-1],[-1,1])
        current_batch_size = size
        true_idx_flattened = tf.range(0,current_batch_size) * NUM_CLASSES + true_index
        true_values = tf.gather(tf.reshape(predicts,[-1]),true_idx_flattened)
        # get neg from non_other part
        top_2_values, top_2_indices = tf.nn.top_k(non_other, 2)
        # other = 0 in true_index
        true_index_other = tf.multiply(true_index, tf.cast(other_flag, tf.int32))
        top_1_flag = tf.nn.in_top_k(non_other, true_index_other, 1)
        # other = False(0) in top_1_flag
        top_1_index = tf.multiply(tf.cast(top_1_flag, tf.int32), tf.cast(other_flag, tf.int32))
        rows = [tf.squeeze(row) for row in tf.split(0,current_batch_size,top_2_indices)]
        idxs = tf.split(0,current_batch_size,top_1_index)
        neg_idx = [tf.gather(rows[i],idxs[i]) for i in range(current_batch_size)]
        neg_idx = tf.concat(0, neg_idx)
        neg_idx = tf.range(0,current_batch_size) * (NUM_CLASSES - 1) + neg_idx
        neg_values = tf.gather(tf.reshape(non_other, [-1]), neg_idx)
        # get other values, non-other = 0
        other_values = tf.multiply(tf.squeeze(other), tf.cast(other_flag, tf.float32))
        return true_values,neg_values,other_values

    # Training computation
    logits, alpha = model(train_data_node, train_pf_node, train_sdp_node)
    predict_scores = tf.nn.softmax(logits)
    train_label = tf.argmax(train_labels_node,1)

    is_train = tf.constant(0.5)
    t_values, n_values, o_values = tf.cond(tf.equal(dropout_keep_prob,is_train),lambda: get_targe_neg(BATCH_SIZE,train_label,predict_scores),
                                 lambda: get_targe_neg(Test_Size,train_label,predict_scores))
    # pairwise ranking loss
    loss = tf.reduce_mean(1.0 - t_values + beta * n_values + o_values)
    # v_c = [num_class,num_class]
    normalized_cm = tf.nn.l2_normalize(classes_matrix, dim=0)
    vertical_constraint = tf.matmul(normalized_cm, normalized_cm, transpose_a=True)
    regularizers_1 = tf.nn.l2_loss(classes_matrix)
    regularizers_2 = tf.reduce_sum(vertical_constraint) - tf.trace(vertical_constraint)
    loss += 0.01 * regularizers_1 + 0.01 * regularizers_2

    tf.scalar_summary('loss', loss)

    # optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    learning_rate = tf.Variable(start_learning_rate,name="learning_rate")
    # learning_rate=tf.train.exponential_decay(start_learning_rate,global_step*BATCH_SIZE,train_size,0.9,staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Evaluate model
    train_predict = tf.argmax(logits,1)
    train_label = tf.argmax(train_labels_node,1)
    # train accuracy
    train_correct_pred = tf.equal(train_predict,train_label)
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))
    tf.scalar_summary('acc', train_accuracy)
    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    def dev_step(x_batch,y_batch,pf_batch,w_batch,best_test_loss,sess):
        feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,train_pf_node:pf_batch,train_sdp_node:w_batch,dropout_keep_prob:1.0}
        # Run the graph and fetch some of the nodes.
        # test dont apply train_op (train_op is update gradient).
        summary,step, losses, lr,acc,y_label,y_predict,test_predicts,dev_alpha = sess.run([merged,global_step, loss,learning_rate,train_accuracy,train_label,train_predict,logits,alpha]
                                                                   ,feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, lr {:g} ,acc {:g}".format(time_str, step, losses,lr,acc))
        # print("{}: step {}, loss {:g} ,acc {:g}".format(time_str, step, losses,acc))
        # compute index
        f1 = official_score.official_score(y_predict)
        if len(record_f1) > 0 and f1 > max(record_f1):
            if os.path.exists('proposed_best_2717.txt'):
                os.remove('proposed_best_2717.txt')
            os.rename('proposed_test_2717.txt','proposed_best_2717.txt')
            # save predicts
            record_predicts.clear()
            record_predicts.append(test_predicts)
            # save alpha
            record_alpha.clear()
            record_alpha.append(dev_alpha)
        else:
            # delete proposed file
            os.remove('proposed_test_2717.txt')

        # put in record
        record_f1.append(f1)
        record_acc.append(acc)

        new_best_test_loss = best_test_loss
        # decide if need to decay learning rate
        if (step % steps_each_check < 100) and (step > 100):
            loss_delta = (best_test_loss if best_test_loss is not None else 0 ) - losses
            if best_test_loss is not None and loss_delta < decay_delta:
                print('validation loss did not improve enough, decay learning rate')
                current_learning_rate = min_learning_rate if lr * learning_rate_decay < min_learning_rate else lr * learning_rate_decay
                if current_learning_rate == min_learning_rate:
                    print('It is already the smallest learning rate.')
                sess.run(learning_rate.assign(current_learning_rate))
                print('new learning rate is: ', current_learning_rate)
            else:
                # update
                new_best_test_loss = losses

        return new_best_test_loss

    # run the training
    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
        tf.initialize_all_variables().run()
        print('Initialized!')
        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train,y_train,pf_train,w_train)),BATCH_SIZE,NUM_EPOCHS)
        # batch count
        batch_count = 0
        best_test_loss = None
        # Training loop.For each batch...
        for batch in batches:
            batch_count += 1
            if batch_count % EVAL_FREQUENCY == 0:
                print("\nEvaluation:")
                best_test_loss=dev_step(x_test,y_test,pf_test,w_test,best_test_loss,sess)
                print("")
            else:
                if  batch_count % META_FREQUENCY == 99:
                    x_batch, y_batch, pf_batch, w_batch = zip(*batch)
                    feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,train_pf_node:pf_batch,train_sdp_node:w_batch,dropout_keep_prob:0.5}
                    # Run the graph and fetch some of the nodes.
                    # option
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _,summary, step, losses, acc = sess.run([train_op,merged,global_step, loss,train_accuracy],
                                                            feed_dict=feed_dict,
                                                            options=run_options,
                                                            run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g},acc {:g}".format(time_str, step, losses,acc))
                else:
                    x_batch, y_batch, pf_batch, w_batch = zip(*batch)
                    feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,train_pf_node:pf_batch,train_sdp_node:w_batch,dropout_keep_prob:0.5}
                    # Run the graph and fetch some of the nodes.
                    _, summary, step, losses, acc = sess.run([train_op,merged,global_step, loss,train_accuracy],feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses,acc))

        # save model
        save_path = saver.save(sess, "model_class_att.ckpt")
        print("Model saved in file: ", save_path)
        # save test predicts
        numpy.save('test_predicts.npy', record_predicts[0])
        # save alpha
        numpy.save('sdp_att_alpha.npy', record_alpha[0])
        # print best result
        best_acc = max(record_acc)
        best_f1 = max([record_f1[i] for i,a in enumerate(record_acc) if a==best_acc])
        print('best acc: '+str(best_acc)+' '+'F1: '+str(best_f1))
        print('Highest F1: '+str(max(record_f1)))
        train_writer.close()
        test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='/tmp/cnn_logs',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()