#!/usr/bin/env python
# _*_ coding: utf-8 _*_
#
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import numpy as np
# import time
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import precision_score, recall_score, f1_score
# import os
# import sys
# sys.path.append('..')
# import config_jxd as config
# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import host_subplot
#
#
#
# # ===========  global parameters  ===========
#
# T = 5  # iteration
# N = 2  # embedding_depth
# P = 64  # embedding_size
# D = 8  # dimensional,feature num
# B = 10  # mini-batch
# lr = 0.0001  # learning_rate
# # MAX_SIZE = 0 # record the max number of a function's block
# max_iter = 10  # 原数据：100
# decay_steps = 10  # 衰减步长
# decay_rate = 0.0001  # 衰减率
# snapshot = 1
# is_debug = True
# n_layer = 3
# block_embedding = 768
# BE = 768
# BED = 776
# IE = 100
# MAX_I = 20
#
#
#
#
#
# # program_name = "coreutils"
# # program_version = "6.12"
# # program_arch = "arm"
# # program_opti = "O0-O1-O2-O3"
#
# # program_name = "openssl"
# # program_version = "1.0.1u"
# # program_arch = "arm"
# # program_opti = "O0-O3"
#
# program_name = "busybox"
# program_version = "1.27.2"
# program_arch = "arm"
# program_opti = "O0-O3"
#
# PREFIX = program_name + '_' + program_version + '_' + program_arch + '_' + program_opti
# TRAIN_TFRECORD = config.TFRECORD_EMBEDDING_DIR_OLD + os.sep + "train_" + PREFIX + ".tfrecord"
# TEST_TFRECORD = config.TFRECORD_EMBEDDING_DIR_OLD + os.sep + "test_" + PREFIX + ".tfrecord"
# VALID_TFRECORD = config.TFRECORD_EMBEDDING_DIR_OLD + os.sep + "valid_" + PREFIX + ".tfrecord"
# program_info = (program_name, program_version, program_arch, program_opti)
#
#
# precision_file = config.FIT_RESULT_DIR + os.sep + PREFIX + "_precision.csv"
# AUC_file = config.FIT_RESULT_DIR + os.sep + PREFIX + "_AUC.csv"
# recall_file = config.FIT_RESULT_DIR + os.sep + PREFIX + "_recall.csv"
# f1score_file = config.FIT_RESULT_DIR + os.sep + PREFIX + "_f1score.csv"
# accuracy_file = config.FIT_RESULT_DIR + os.sep + PREFIX + "_accuarcy.csv"
#
#
# train_label_predict_file = config.FIT_RESULT_DIR + os.sep + PREFIX + "_train_label_predict_r.csv"
# test_label_predict_file = config.FIT_RESULT_DIR + os.sep + PREFIX + "_test_label_predict_r.csv"
#
#
# csv_file = config.DATASET_DIR
# # result_file = config.ZTHtest + os.sep + "test" + str(config.TRAIN_DATASET_NUM) + "_[" + '_'.join(
# #     config.STEP3_PORGRAM_ARR) + "].txt"
#
# # result_fp = open(result_file, "w")
# # =============== convert the real data to training data ==============
# #       1.  construct_learning_dataset() combine the dataset list & real data
# #       1-1. generate_adj_matrix_pairs()    traversal list and construct all the matrixs
# #       1-1-1. convert_graph_to_adj_matrix()    process each cfg
# #       1-2. generate_features_pair() traversal list and construct all functions' feature map
# # =====================================================================
#
# def block_embed(X_inst_embed,max_inst_num,inst_dim,n_hidden):
#
#
#     X = tf.reshape(X_inst_embed,[-1,max_inst_num,inst_dim])
#     X = tf.transpose(X,[1,0,2])
#     X = tf.reshape(X,[-1,inst_dim])
#     X = tf.split(X,max_inst_num,0)
#     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
#     outputs ,states = tf.nn.static_rnn(lstm_cell,X,dtype = tf.float32)
#     ret = tf.reshape(outputs[-1],[B,-1,n_hidden])
#     return ret
#
#
#
# def structure2vec_net(cdfgs, block_eb, inst_eb):
#     """
#     iput: [B,None,None] [B,None,D]
#
#     """
#     with tf.variable_scope("structure2vec_net") as structure2vec_net:
#         B_mu_5 = tf.Variable(tf.zeros(shape=[0, P]), trainable=False)
#         weights = {}
#         # w_1 = tf.get_variable('w_1', [D + BE + IE, P], tf.float32,
#         #                       tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         # w_1 = tf.get_variable('w_1', [D + BE , P], tf.float32,
#         # w_1 = tf.get_variable('w_1', [IE, P], tf.float32,
#         #                       tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         w_1 = tf.get_variable('w_1', [BE, P], tf.float32,
#                               tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         w_2 = tf.get_variable('w_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         for i in range(3):
#             weights['w1%d' % (i + 1)] = tf.get_variable('w1%d' % (i + 1), [P, P], tf.float32,
#                                                         tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#             weights['w2%d' % (i + 1)] = tf.get_variable('w2%d' % (i + 1), [P, P], tf.float32,
#                                                         tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         # w2v_lstm = block_embed(inst_eb,MAX_I,IE,IE)
#         # x = tf.concat([block_eb,w2v_lstm],axis=0)
#         x = block_eb
#         # x = w2v_lstm
#         node_embeddings = tf.einsum('abc,cd->abd', x, w_1)
#         # node_embeddings = tf.einsum('abc,cd->abd', w2v_lstm, w_1)# [B, max_node, P]
#         cur_node_embeddings = tf.nn.relu(node_embeddings)
#         # propagation layer
#         # cdfgs [B, max_node,max_node]
#         for i in range(n_layer):
#             cur_node_embeddings = tf.matmul(cdfgs, cur_node_embeddings)  # message aggregation
#             cur_node_embeddings = tf.einsum('abc,cd->abd', cur_node_embeddings, weights['w1%d' % (i + 1)])
#             if i + 1 < n_layer:
#                 cur_node_embeddings = tf.nn.relu(cur_node_embeddings)
#             tot_node_embeddings = cur_node_embeddings + node_embeddings
#             cur_node_embeddings = tf.nn.tanh(tot_node_embeddings)
#
#         g_embed = tf.reduce_sum(cur_node_embeddings, 1)  # [batch, P]
#         output = tf.matmul(g_embed, w_2)
#         return output
#
#
#
# def bi_rnn(X_inst_embed,max_inst_num,inst_dim):
#     X = tf.reshape(X_inst_embed,[-1,max_inst_num,inst_dim])
#     X = tf.transpose(X,[1,0,2])
#     X = tf.reshape(X,[-1,inst_dim])  #(batch*block_num*max_inst_num,inst_dim)
#     X = tf.split(X, max_inst_num, 0) # get a list of max_inst_num * (batch * block_num , inst_dim)
#     forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = inst_dim)
#     backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = inst_dim)
#     bi_rnn_output , _ , _ = tf.nn.static_bidirectional_rnn(forward_cell,backward_cell,X,dtype=tf.float32)
#     ret = tf.reshape(tf.stack(bi_rnn_output), [B,-1,max_inst_num,2*inst_dim])
#     return ret
#
#
# def safe_net(inst_eb):
#     with tf.variable_scope("safe_net") as safe_net:
#         r = 10
#         da = 100
#         e = 200
#         n = P
#         ws1 = tf.get_variable('ws1',[da,2*IE],tf.float32,tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         ws2 = tf.get_variable('ws2',[r,da],tf.float32,tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         H = bi_rnn(inst_eb,MAX_I,IE)#[Batch,block,MAX_I,2*IE] , [Batch,block,m,u]
#         HT = tf.transpose(H,[0,1,3,2]) #HT [Batch,block,2*IE,MAX_I]
#         w1_HT = tf.einsum('ec,abcd->abed',ws1,HT) #[batch,block,da,MAX_I],which is [batch,block,da,m]
#         tanh = tf.nn.tanh(w1_HT)
#         w2_tanh = tf.einsum('ec,abcd->abed',ws2,tanh)#[batch,block,r,m]
#         matrix_A = tf.nn.softmax(w2_tanh)
#         matrix_B = tf.einsum('abcd,abde->abce',matrix_A,H)#A*H->[batch,block,r,u],[batch,block,r,MAX_I]
#         flatten_B = tf.reshape(matrix_B,[B,-1,r * MAX_I])
#         reduce_sum_B = tf.reduce_sum(flatten_B, 1)  #[batch,r*u]
#         # weight_l1 = tf.get_variable('weight_l1',[r*MAX_I,r * MAX_I],tf.float32,
#         #                          tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
#         # weight_l2 = tf.get_variable('weight_l2', [r * MAX_I, r * MAX_I], tf.float32,
#         #                             tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#
#         layer0 = tf.layers.dense(reduce_sum_B,r*MAX_I)
#         # layer1 = tf.layers.dense(layer0,r*MAX_I)
#         out_W1 = tf.get_variable('out_W1',[e,r * MAX_I],tf.float32,
#                                  tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
#         out_W2 = tf.get_variable('out_W2', [n,e], tf.float32,
#                                  tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         mid1 = tf.einsum('ab,cb->ca',out_W1,layer0) #[batch , e]
#         mid2 = tf.nn.relu(mid1)
#         out = tf.einsum('ab,cb->ca',out_W2,mid2) # [batch,n]
#         return out
#
#
# # def calculate_auc(labels, predicts):
# #     fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=1)
# #     AUC = auc(fpr, tpr)
# #     print "auc : ", AUC
# #     return AUC
# def calculate_auc(labels, predicts):
#     fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=1)
#
#     new_predicts = []
#
#     for i in range(len(predicts)):
#         if labels[i] == 1:
#             if predicts[i] > 0.8:
#                 new_predicts.append(labels[i])
#             else:
#                 new_predicts.append(-1)
#
#         if labels[i] == -1:
#             if predicts[i] < 0.8:
#                 new_predicts.append(labels[i])
#             else:
#                 new_predicts.append(1)
#
#     precision = precision_score(labels, new_predicts, pos_label=1)
#     recall = recall_score(labels, new_predicts)
#     f1 = f1_score(labels, new_predicts)
#
#     AUC = auc(fpr, tpr)
#     print ("auc : ", AUC)
#     return fpr, tpr, AUC, precision, recall, f1
#
# def contrastive_loss(labels, distance):
#     #    tmp= y * tf.square(d)
#     #    #tmp= tf.mul(y,tf.square(d))
#     #    tmp2 = (1-y) * tf.square(tf.maximum((1 - d),0))
#     #    return tf.reduce_sum(tmp +tmp2)/B/2
#     #    print "contrastive_loss", y,
#     loss = tf.to_float(tf.reduce_sum(tf.square(distance - labels)))
#     return loss
#
#
# '''
# 张天豪修改　　　１１.４
# '''
#
#
# def get_positive_expectation(p_samples, average=True):
#
#     log_2 = math.log(2.)
#     Ep = log_2 - tf.nn.softplus(- p_samples)
#
#     return Ep
#
#
# '''
# 张天豪修改　　　１１．４
# '''
#
#
# def get_negative_expectation(q_samples, average=True):
#     """Computes the negative part of a JS Divergence.
#     Args:
#         q_samples: Negative samples.
#         average: Average the result over samples.
#     Returns:
#         th.Tensor
#     """
#     log_2 = math.log(2.)
#     Eq = tf.nn.softplus(-q_samples) + q_samples - log_2
#
#     return Eq
#
#
# def cal_anotherloss(model1, model2, node1, node2, num1, num2):
#     pos_MI = tf.constant([0], dtype=tf.float32)  # 正例
#     neg_MI = tf.constant([0], dtype=tf.float32)  # 负例
#     loss_list = []
#     for i in range(5):
#         graph_vec1 = tf.slice(model1, [i, 0], [1, P])  # 截取图嵌入
#         graph_vec2 = tf.slice(model2, [i, 0], [1, P])
#         MI_ori1 = tf.matmul(node1[i], tf.transpose(graph_vec2))  # 节点嵌入和图嵌入点积
#         MI_ori2 = tf.matmul(node2[i], tf.transpose(graph_vec1))
#
#         '''
#         张天豪修改　　　　１１．４
#         '''
#         MI_ori1 = get_positive_expectation(MI_ori1)
#         MI_ori2 = get_positive_expectation(MI_ori2)
#
#         MI_sum1 = tf.reduce_sum(MI_ori1, axis=0)  # 结果相加
#         MI_sum2 = tf.reduce_sum(MI_ori2, axis=0)
#
#         condition = tf.less(num1[i], num2[i])
#
#         node_num = tf.where(condition, num2[i], num1[i])  # 选出最大的节点数
#
#         MI1 = tf.divide(MI_sum1, node_num)  # 除以最大的节点数
#         MI2 = tf.divide(MI_sum2, node_num)
#
#         MI_fin = tf.add(MI1, MI2)
#         pos_MI = tf.add(MI_fin, pos_MI)  # 5个正例子加一起
#
#     for i in range(5, 10):
#         graph_vec1 = tf.slice(model1, [i, 0], [1, P])
#         graph_vec2 = tf.slice(model2, [i, 0], [1, P])
#         MI_ori1 = tf.matmul(node1[i], tf.transpose(graph_vec2))
#         MI_ori2 = tf.matmul(node2[i], tf.transpose(graph_vec1))
#
#         MI_ori1 = get_negative_expectation(MI_ori1)
#         MI_ori2 = get_negative_expectation(MI_ori2)
#
#         MI_sum1 = tf.reduce_sum(MI_ori1, axis=0)
#         MI_sum2 = tf.reduce_sum(MI_ori2, axis=0)
#
#         condition = tf.less(num1[i], num2[i])
#
#         node_num = tf.where(condition, num2[i], num1[i])
#
#         MI1 = tf.divide(MI_sum1, node_num)
#         MI2 = tf.divide(MI_sum2, node_num)
#
#         MI_fin = tf.add(MI1, MI2)
#         neg_MI = tf.add(MI_fin, neg_MI)
#
#     pos_loss = tf.divide(pos_MI, 5.0)
#     neg_loss = tf.divide(neg_MI, 5.0)
#
#     loss_list.append(pos_loss)
#     loss_list.append(neg_loss)
#
#     return neg_loss - pos_loss
#
#
# def cal_finloss(loss, another_loss):
#     '''
#     张天豪修改　　　　　１１．４
#     '''
#
#     W_3 = tf.get_variable('W_3', [1], tf.float32, initializer=tf.constant_initializer(0.001))
#
#     finloss = tf.add(loss, (W_3 * another_loss))
#
#     return finloss
#
#
# def plot_acc_loss(name, loss, acc):
#     host = host_subplot(111)  # row=1 col=1 first pic
#     plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
#     par1 = host.twinx()  # 共享x轴
#
#     # set labels
#     host.set_xlabel("steps")
#     host.set_ylabel(name + "-loss")
#     par1.set_ylabel(name + "-accuracy")
#
#     # plot curves
#     p1, = host.plot(range(len(loss)), loss, label="loss")
#     p2, = par1.plot(range(len(acc)), acc, label="accuracy")
#
#     # set location of the legend,
#     # 1->rightup corner, 2->leftup corner, 3->leftdown corner
#     # 4->rightdown corner, 5->rightmid ...
#     host.legend(loc=5)
#
#     # set label color
#     host.axis["left"].label.set_color(p1.get_color())
#     par1.axis["right"].label.set_color(p2.get_color())
#
#     # set the range of x axis of host and y axis of par1
#     # host.set_xlim([-200, 5200])
#     # par1.set_ylim([-0.1, 1.1])
#
#     plt.draw()
#     plt.show()
#
#
# def compute_accuracy(prediction, labels):
#     accu = 0.0
#     threshold = 0.5
#     for i in range(len(prediction)):
#         if labels[i][0] == 1:
#             if prediction[i][0] > threshold:
#                 accu += 1.0
#         else:
#             if prediction[i][0] < threshold:
#                 accu += 1.0
#     acc = accu / len(prediction)
#     return acc
#
#
# def cal_distance(model1, model2):
#     a_b = tf.reduce_sum(tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(model1, (1, -1)),
#                                                              tf.reshape(model2, (1, -1))], 0), 0), (B, P)), 1,
#                         keep_dims=True)
#     a_norm = tf.sqrt(tf.reduce_sum(tf.square(model1), 1, keep_dims=True))
#     b_norm = tf.sqrt(tf.reduce_sum(tf.square(model2), 1, keep_dims=True))
#     distance = a_b / tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(a_norm, (1, -1)),
#                                                           tf.reshape(b_norm, (1, -1))], 0), 0), (B, 1))
#     return distance
#
#
# def read_and_decode(filename):
#     # 根据文件名生成一个队列
#     filename_queue = tf.train.string_input_producer([filename])
#     # create a reader from file queue
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     # get feature from serialized example
#
#     features = tf.parse_single_example(serialized_example, features={
#         'label': tf.FixedLenFeature([], tf.int64),
#         'cfg_1': tf.FixedLenFeature([], tf.string),
#         'cfg_2': tf.FixedLenFeature([], tf.string),
#         'func_fea_1': tf.FixedLenFeature([], tf.string),
#         'func_fea_2': tf.FixedLenFeature([], tf.string),
#         'node_fea_1': tf.FixedLenFeature([], tf.string),
#         'node_fea_2': tf.FixedLenFeature([], tf.string),
#         'num1': tf.FixedLenFeature([], tf.int64),
#         'num2': tf.FixedLenFeature([], tf.int64),
#         'max': tf.FixedLenFeature([], tf.int64),
#         'inst_embedding1': tf.FixedLenFeature([], tf.string),
#         'inst_embedding2': tf.FixedLenFeature([], tf.string),
#         'inst_num1': tf.FixedLenFeature([], tf.string),
#         'inst_num2': tf.FixedLenFeature([], tf.string),
#         'w2v_1': tf.FixedLenFeature([], tf.string),
#         'w2v_2': tf.FixedLenFeature([], tf.string),
#         'w2v_num_1': tf.FixedLenFeature([], tf.string),
#         'w2v_num_2': tf.FixedLenFeature([], tf.string)
#     })
#
#
#     label = tf.cast(features['label'], tf.int32)
#
#     cfg_1 = features['cfg_1']
#     cfg_2 = features['cfg_2']
#     # result_fp.write(str(cfg_1))
#     # result_fp.write(str(cfg_2))
#
#
#     func_fea_1 = features['func_fea_1']
#     func_fea_2 = features['func_fea_2']
#
#     num1 = tf.cast(features['num1'], tf.int32)
#     node_fea_1 = features['node_fea_1']
#
#     num2 = tf.cast(features['num2'], tf.int32)
#     node_fea_2 = features['node_fea_2']
#     max_num = tf.cast(features['max'], tf.int32)
#
#     block_embedding1 = features['inst_embedding1']
#     block_embedding2 = features['inst_embedding2']
#     block_num1 = features['inst_num1']
#     block_num2 = features['inst_num2']
#     inst_embedding1 = features['w2v_1']
#     inst_embedding2 = features['w2v_2']
#     #
#     inst_num1 = features['w2v_num_1']
#     inst_num2 = features['w2v_num_2']
#     return label, cfg_1, cfg_2, func_fea_1, func_fea_2, node_fea_1, node_fea_2, num1, num2, max_num, \
#     block_embedding1, block_embedding2, inst_embedding1, inst_embedding2, inst_num1, inst_num2
#
# def cut_w2v(w2v,maxb,maxi):
#
#     w2v_ori = w2v
#     if w2v_ori.shape[1]>maxi:
#         w2v_ori = w2v_ori[:,:maxi,]
#
#     if w2v_ori.shape[1]<maxi:
#         zero_i = np.zeros([w2v_ori.shape[0],maxi - w2v_ori.shape[1],IE])
#         w2v_ori = np.concatenate([w2v_ori, zero_i], axis=1)
#
#     if w2v_ori.shape[0]>maxb:
#         w2v_ori = w2v_ori[:maxb]
#
#     if w2v_ori.shape[0]<maxb:
#         zero_b = np.zeros([maxb-w2v_ori.shape[0],MAX_I,IE])
#         w2v_ori = np.concatenate([w2v_ori, zero_b], axis=0)
#
#     return w2v_ori
#
# def get_batch(label, cfg_str1, cfg_str2, dfg_str1, dfg_str2, fea_str1, fea_str2, num1, num2, max_num, blockembed1,
#               blockembed2,instembed1, instembed2, instnum1, instnum2):
#     y = np.reshape(label, [B, 1])
#
#     v_num_1 = []
#     v_num_2 = []
#     for i in range(B):
#         v_num_1.append([int(num1[i])])
#         v_num_2.append([int(num2[i])])
#
#     # 补齐 martix 矩阵的长度
#     # cdfg_1 = []
#     # cdfg_2 = []
#     cfg_1 = []
#     cfg_2 = []
#     for i in range(B):
#         cfg_arr = np.array(cfg_str1[i].decode().split(','))
#         cfg_adj = np.reshape(cfg_arr, (int(num1[i]), int(num1[i])))  # reshape成邻接矩阵
#         cfg_ori1 = cfg_adj.astype(np.float32)
#         for node in range(int(num1[i])):
#             cfg_ori1[node][node] = 1.
#         # cfg only
#         cfg_zero1 = np.zeros([int(num1[i]), (int(max_num[i]) - int(num1[i]))])
#         cfg_zero2 = np.zeros([(int(max_num[i]) - int(num1[i])), int(max_num[i])])
#         cfg_vec1 = np.concatenate([cfg_ori1, cfg_zero1], axis=1)
#         cfg_vec2 = np.concatenate([cfg_vec1, cfg_zero2], axis=0)
#         cfg_1.append(cfg_vec2.tolist())
#
#         cfg_arr = np.array(cfg_str2[i].decode().split(','))
#         cfg_adj = np.reshape(cfg_arr, (int(num2[i]), int(num2[i])))
#         cfg_ori2 = cfg_adj.astype(np.float32)
#         for node in range(int(num2[i])):
#             cfg_ori2[node][node] = 1.
#
#         cfg_zero1 = np.zeros([int(num2[i]), (int(max_num[i]) - int(num2[i]))])
#         cfg_zero2 = np.zeros([(int(max_num[i]) - int(num2[i])), int(max_num[i])])
#         cfg_vec1 = np.concatenate([cfg_ori2, cfg_zero1], axis=1)
#         cfg_vec2 = np.concatenate([cfg_vec1, cfg_zero2], axis=0)
#         cfg_2.append(cfg_vec2.tolist())
#     # 补齐 feature 列表的长度
#     bert_1 = []
#     bert_2 = []
#     w2v_1 = []
#     w2v_2 = []
#     max_instnum1 = 0
#     for blk in instnum1:
#         for inst in np.array(blk.decode().split(',')).astype(np.int32):
#             if inst > max_instnum1:
#                 max_instnum1 = inst
#
#     max_instnum2 = 0
#     for blk in instnum2:
#         for inst in np.array(blk.decode().split(',')).astype(np.int32):
#             if inst > max_instnum2:
#                 max_instnum2 = inst
#
#     for i in range(B):
#         # fea_arr = np.array(fea_str1[i].split(','))
#         # fea_ori = fea_arr.astype(np.float32)
#         # fea_ori1 = np.resize(fea_ori, (int(num1[i]), D))
#
#         inst_embed_arr1 = np.array(instembed1[i].decode().split(','))
#         inst_embed_ori1 = inst_embed_arr1.astype(np.float32)
#         inst_embed_vec1 = np.reshape(inst_embed_ori1, [-1, IE])
#         inst_num_arr1 = np.array(instnum1[i].decode().split(','))
#         inst_num_ori1 = inst_num_arr1.astype(np.int64)
#         inst_func1 = []
#
#         flag1 = 0
#         for inst_n in inst_num_ori1:
#             # inst_func1.append(inst_embed_vec1[flag1:inst_n + flag1])
#             # flag1 = flag1 + inst_n
#             w2v_block_ori1 = inst_embed_vec1[flag1:inst_n + flag1]
#             inst_zero1 = np.zeros([int(max_instnum1 - inst_n), IE])
#             w2v_block_vec1 = np.concatenate([w2v_block_ori1, inst_zero1], axis=0)
#             inst_func1.append(w2v_block_vec1)
#             flag1 = flag1 + inst_n
#
#         inst_func1 = np.array(inst_func1)
#         block_zero1 = np.zeros([int(max_num[i] - num1[i]), int(max_instnum1), IE])
#         w2v_func_vec1 = np.concatenate([inst_func1,block_zero1])
#
#
#         w2v_1.append(cut_w2v(w2v_func_vec1,max_num[i],MAX_I).astype(np.float32))
#
#         block_embed_arr1 = np.array(blockembed1[i].decode().split(','))
#         block_embed_ori1 = block_embed_arr1.astype(np.float32)
#         block_embed_vec1 = np.reshape(block_embed_ori1, [-1, BE])
#
#         # fea_block_1 = np.concatenate([fea_ori1, block_embed_vec1], axis=1)
#         # fea_zero1 = np.zeros([int(max_num[i] - num1[i]), D + BE])
#         bert_zero1 = np.zeros([int(max_num[i] - block_embed_vec1.shape[0]),BE])
#         # fea_vec1 = np.concatenate([fea_block_1, fea_zero1], axis=0)
#         bert_vec1 = np.concatenate([block_embed_vec1, bert_zero1], axis=0)
#         bert_1.append(bert_vec1)
#
#         # fea_arr = np.array(fea_str2[i].split(','))
#         # fea_ori = fea_arr.astype(np.float32)
#         # fea_ori2 = np.resize(fea_ori, (int(num2[i]), D))
#
#         inst_embed_arr2 = np.array(instembed2[i].decode().split(','))
#         inst_embed_ori2 = inst_embed_arr2.astype(np.float32)
#         inst_embed_vec2 = np.reshape(inst_embed_ori2, [-1, IE])
#         inst_num_arr2 = np.array(instnum2[i].decode().split(','))
#         inst_num_ori2 = inst_num_arr2.astype(np.int64)
#         inst_func2 = []
#
#         flag2 = 0
#         for inst_n in inst_num_ori2:
#             # inst_func1.append(inst_embed_vec1[flag1:inst_n + flag1])
#             # flag1 = flag1 + inst_n
#             w2v_block_ori2 = inst_embed_vec2[flag2:inst_n + flag2]
#             inst_zero2 = np.zeros([int(max_instnum2 - inst_n), IE])
#             w2v_block_vec2 = np.concatenate([w2v_block_ori2, inst_zero2], axis=0)
#             inst_func2.append(w2v_block_vec2)
#             flag2 = flag2 + inst_n
#         inst_func2 = np.array(inst_func2)
#         block_zero2 = np.zeros([int(max_num[i] - num2[i]), int(max_instnum2), IE])
#         w2v_func_vec2 = np.concatenate([inst_func2,block_zero2])
#         w2v_2.append(cut_w2v(w2v_func_vec2,max_num[i],MAX_I).astype(np.float32))
#         # a = block_embed(w2v_2[0], 586, 100, 100)
#
#         block_embed_arr2 = np.array(blockembed2[i].decode().split(','))
#         block_embed_ori2 = block_embed_arr2.astype(np.float32)
#         block_embed_vec2 = np.reshape(block_embed_ori2, [-1, BE])
#         bert_zero2 = np.zeros([int(max_num[i] - block_embed_vec2.shape[0]), BE])
#         bert_vec2 = np.concatenate([block_embed_vec2, bert_zero2], axis=0)
#         bert_2.append(bert_vec2)
#
#     # a = block_embed(w2v_1[0][0], len(w2v_1[0][0]), 100, 100)
#
#     return y, cfg_1, cfg_2, bert_1, bert_2, v_num_1, v_num_2, w2v_1, w2v_2,max_instnum1,max_instnum2,max_num[0]#max_num is the same,which is 586
#
# def com_label_predict(labels,predicts,lables_predicts_file):
#     labels_r = np.reshape(labels, (-1))
#     predicts_r = np.reshape(predicts, (-1))
#
#     with open(lables_predicts_file, "w") as lpf:
#
#         for i in range(len(predicts_r)):
#             label = labels_r[i]
#
#             predict = predicts_r[i]
#
#             res_str = str(predict) +','+ str(label) + '\n'
#
#             lpf.write(res_str)
#
#     lpf.close()
#
#
# def write_results(csv_file, result_list):
#     with open(csv_file, "w") as fp:
#         for e in result_list:
#             res_str = str(e) + '\n'
#             fp.write(res_str)
#
#     fp.close()
#
# def get_datasetnum(csv_file, program_info):
#
#     test_dataset_num = 0
#     train_dataset_num = 0
#     valid_dataset_num = 0
#
#     for csv_name in os.listdir(csv_file):
#         csv_name_list = csv_name.split('_')
#         csv_info = (csv_name_list[1], csv_name_list[2], csv_name_list[3], csv_name_list[4])
#         if csv_name_list[0] == "test" and csv_info == program_info:
#             with open(csv_file + os.sep + csv_name, 'r') as f:
#                 test_dataset_num = len(f.readlines())
#
#         if csv_name_list[0] == "train" and csv_info == program_info:
#             with open(csv_file + os.sep + csv_name, 'r') as f:
#                 train_dataset_num = len(f.readlines())
#
#         if csv_name_list[0] == "valid" and csv_info == program_info:
#             with open(csv_file + os.sep + csv_name, 'r') as f:
#                 valid_dataset_num = len(f.readlines())
#     # print 'asdfg'
#     return train_dataset_num, valid_dataset_num, test_dataset_num
# # 4.construct the network
# # Initializing the variables
# # Siamese network major part
#
# # Initializing the variables
#
# init = tf.global_variables_initializer()
# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase=True)
#
# v_num_left = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_left')
# cdfg_left = tf.placeholder(tf.float32, shape=([B, None, None]), name='cdfg_left')
# # fea_left = tf.placeholder(tf.float32, shape=([B, None, D + BE + IE]), name='fea_left')
# fea_left = tf.placeholder(tf.float32, shape=([B, None, BE ]), name='fea_left')
# w2v_left = tf.placeholder(tf.float32, shape=([B, None, None, IE ]), name='w2v_left')
# inst_max_left = tf.placeholder('int32')
#
# v_num_right = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_right')
# cdfg_right = tf.placeholder(tf.float32, shape=([B, None, None]), name='cdfg_right')
# # fea_right = tf.placeholder(tf.float32, shape=([B, None, D + BE + IE]), name='fea_right')
# fea_right = tf.placeholder(tf.float32, shape=([B, None, BE ]), name='fea_right')
# w2v_right = tf.placeholder(tf.float32, shape=([B, None, None,IE ]), name='w2v_right')
# inst_max_right = tf.placeholder('int32')
#
# max_num = tf.placeholder('int32')
# labels = tf.placeholder(tf.float32, shape=([B, 1]), name='gt')
#
# dropout_f = tf.placeholder("float")
#
#
# with tf.variable_scope("siamese") as siamese:
#
#
#     model1 = structure2vec_net(cdfg_left, fea_left, w2v_left)#(cdfgs, block_eb, inst_eb,max_num,max_inst_num)
#     siamese.reuse_variables()
#     model2 = structure2vec_net(cdfg_right, fea_right,w2v_right)
#     # model1 = safe_net(w2v_left)
#     # siamese.reuse_variables()
#     # model2 = safe_net(w2v_right)
#
# dis = cal_distance(model1, model2)
#
# loss = contrastive_loss(labels, dis)
#
# # another_loss = cal_anotherloss(model1,model2,node1,node2,num1,num2)
#
# # finloss = cal_finloss(loss,another_loss)
#
#
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
# # 三大数据：基本块特征，控制流图，数据流图
# # list_train_label：
# # cfg :控制流图 dfg:数据流图  fea:基本块特征
# ''
#
# list_train_label, list_train_cfg_1, list_train_cfg_2, list_train_func_fea_1, list_train_func_fea_2, list_train_node_fea_1, \
# list_train_node_fea_2, list_train_num1, list_train_num2, list_train_max, \
# list_train_block_embedding1, list_train_block_embedding2, \
# list_train_inst_embedding1, list_train_inst_embedding2, list_train_inst_num1, list_train_inst_num2 \
#     = read_and_decode(TRAIN_TFRECORD)
#
# batch_train_label, batch_train_cfg_1, batch_train_cfg_2, batch_train_func_fea_1, batch_train_func_fea_2, batch_train_node_fea_1, \
# batch_train_node_fea_2, batch_train_num1, batch_train_num2, batch_train_max, \
# batch_train_block_embedding1, batch_train_block_embedding2, \
# batch_train_inst_embedding1, batch_train_inst_embedding2, \
# batch_train_inst_num1, batch_train_inst_num2 \
#     = tf.train.batch(
#     [list_train_label, list_train_cfg_1, list_train_cfg_2, list_train_func_fea_1, list_train_func_fea_2,
#      list_train_node_fea_1, list_train_node_fea_2, list_train_num1, list_train_num2, list_train_max,
#      list_train_block_embedding1, list_train_block_embedding2,
#      list_train_inst_embedding1, list_train_inst_embedding2, list_train_inst_num1, list_train_inst_num2],
#     batch_size=B, capacity=10)
#
# list_valid_label, list_valid_cfg_1, list_valid_cfg_2, list_valid_dfg_1, list_valid_dfg_2, list_valid_fea_1, \
# list_valid_fea_2, list_valid_num1, list_valid_num2, list_valid_max, \
# list_valid_block_embedding1, list_valid_block_embedding2, \
# list_valid_inst_embedding1, list_valid_inst_embedding2, list_valid_inst_num1, list_valid_inst_num2 \
#     = read_and_decode(VALID_TFRECORD)
#
# batch_valid_label, batch_valid_cfg_1, batch_valid_cfg_2, batch_valid_dfg_1, batch_valid_dfg_2, batch_valid_fea_1, \
# batch_valid_fea_2, batch_valid_num1, batch_valid_num2, batch_valid_max, \
# batch_valid_block_embedding1, batch_valid_block_embedding2, \
# batch_valid_inst_embedding1, batch_valid_inst_embedding2, batch_valid_inst_num1, batch_valid_inst_num2 \
#     = tf.train.batch([list_valid_label, list_valid_cfg_1, list_valid_cfg_2, list_valid_dfg_1, list_valid_dfg_2,
#                       list_valid_fea_1, list_valid_fea_2, list_valid_num1, list_valid_num2, list_valid_max,
#                       list_valid_block_embedding1, list_valid_block_embedding2,
#                       list_valid_inst_embedding1, list_valid_inst_embedding2, list_valid_inst_num1,
#                       list_valid_inst_num2],
#                      batch_size=B, capacity=10)
#
# list_test_label, list_test_cfg_1, list_test_cfg_2, list_test_dfg_1, list_test_dfg_2, list_test_fea_1, \
# list_test_fea_2, list_test_num1, list_test_num2, list_test_max, \
# list_test_block_embedding1, list_test_block_embedding2, \
# list_test_inst_embedding1, list_test_inst_embedding2, list_test_test_num1, list_test_test_num2 \
#     = read_and_decode(TEST_TFRECORD)
#
# batch_test_label, batch_test_cfg_1, batch_test_cfg_2, batch_test_dfg_1, batch_test_dfg_2, batch_test_fea_1, \
# batch_test_fea_2, batch_test_num1, batch_test_num2, batch_test_max, \
# batch_test_block_embedding1, batch_test_block_embedding2, \
# batch_test_inst_embedding1, batch_test_inst_embedding2, batch_test_inst_num1, batch_test_inst_num2 \
#     = tf.train.batch([list_test_label, list_test_cfg_1, list_test_cfg_2, list_test_dfg_1, list_test_dfg_2,
#                       list_test_fea_1, list_test_fea_2, list_test_num1, list_test_num2, list_test_max,
#                       list_test_block_embedding1, list_test_block_embedding2,
#                       list_test_inst_embedding1, list_test_inst_embedding2, list_test_test_num1, list_test_test_num2],
#                      batch_size=B, capacity=10)
#
# ''''''
# init_opt = tf.global_variables_initializer()
# saver = tf.train.Saver()
# #
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# # config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
# tfconfig = tf.ConfigProto()
# tfconfig.gpu_options.allow_growth = True
# # with tf.Session(config=config) as sess:
#
# train_acc_list = []
# train_loss_list = []
# test_acc_list = []
# test_loss_list = []
#
# with tf.Session(config=tfconfig) as sess:
#     # with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:
#     # writer = tf.summary.FileWriter('logs/', sess.graph)
#     sess.run(init_opt)
#     # flag = 1
#     # for example in tf.python_io.tf_record_iterator(TRAIN_TFRECORD):
#     #     if flag == 895:
#     #         print flag
#     #         x = str(tf.train.Example.FromString(example))
#     #         result_fp.write("zheshiyigehanshu" + str(flag) + x)
#     #         result_fp.write("\n")
#     #         break
#     #     else:
#     #         flag = flag + 1
#     train_num, valid_num, test_num = get_datasetnum(csv_file, program_info)
#     precision_list = []
#     AUC_list = []
#     recall_list = []
#     accuracy_list = []
#     f1score_list = []
#
#     train_label_list = []
#     train_predict_list = []
#     test_label_list = []
#     test_predict_list = []
#     #
#     #     flag = flag + 1
#     # if config.SETP5_IF_RESTORE_VULSEEKER_MODEL:
#     #     saver.restore(sess, config.MODEL_VULSEEKER_DIR + os.sep + config.STEP5_VULSEEKER_MODEL_TO_RESTORE)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     # Training cycle
#     iter = 0
#     while iter < max_iter:
#         iter += 1
#         avg_loss = 0.
#         avg_acc = 0.
#         total_batch = int(train_num / B)
#         start_time = time.time()
#         # Loop over all batches
#         # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_nu
#         print (total_batch)
#         for i in range(total_batch):  # 原数据：total_batch
#
#             train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1, train_node_fea_2, \
#             train_num1, train_num2, train_max, \
#             train_block_embedding1, train_block_embedding2, \
#             train_inst_embedding1, train_inst_embedding2, train_inst_num1, train_inst_num2 \
#                 = sess.run([batch_train_label, batch_train_cfg_1, batch_train_cfg_2, batch_train_func_fea_1,
#                             batch_train_func_fea_2, batch_train_node_fea_1, batch_train_node_fea_2, batch_train_num1,
#                             batch_train_num2, batch_train_max,
#                             batch_train_block_embedding1, batch_train_block_embedding2,
#                             batch_train_inst_embedding1, batch_train_inst_embedding2,
#                             batch_train_inst_num1, batch_train_inst_num2])
#
#             y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2, w2v_1, w2v_2,inst_max1,inst_max2,max_block \
#                 = get_batch(train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1,
#                             train_node_fea_2,
#                             train_num1, train_num2, train_max,
#                             train_block_embedding1, train_block_embedding2,
#                             train_inst_embedding1, train_inst_embedding2, train_inst_num1, train_inst_num2)
#             # MAX_B = max_block
#             # MAX_I1 = inst_max1
#             # MAX_I2 = inst_max2
#             _, loss_value, predict = sess.run([optimizer, loss, dis], feed_dict={
#                 w2v_left: w2v_1, cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1,
#                 w2v_right: w2v_2, cdfg_right: cdfg_2, fea_right: fea_2,v_num_right: v_num_2,
#                 max_num:max_block,
#                 labels: y, dropout_f: 0.9})
#             tr_acc = compute_accuracy(predict, y)
#             # train_acc_list.append(tr_acc)
#             # train_loss_list.append(loss_value)
#             if is_debug:
#                 print ('     %d    tr_acc %0.2f    tr_loss  %0.2f' % (i, tr_acc, loss_value))
#                 # sys.stdout.flush()
#             avg_loss += loss_value
#             avg_acc += tr_acc * 100
#         duration = time.time() - start_time
#
#         if iter % snapshot == 0:
#             # validing model
#             avg_loss = 0.
#             avg_acc = 0.
#             valid_start_time = time.time()
#             print (int(valid_num / B))
#             for m in range(int(valid_num / B)):  # 原数据：int(valid_num / B)
#                 valid_label, valid_cfg_1, valid_cfg_2, valid_func_fea_1, valid_func_fea_2, valid_node_fea_1, valid_node_fea_2, \
#                 valid_num1, valid_num2, valid_max, \
#                 valid_block_embedding1, valid_block_embedding2, \
#                 valid_inst_embedding1, valid_inst_embedding2, valid_inst_num1, valid_inst_num2 \
#                     = sess.run([batch_valid_label, batch_valid_cfg_1, batch_valid_cfg_2, batch_valid_dfg_1,
#                                 batch_valid_dfg_2, batch_valid_fea_1, batch_valid_fea_2, batch_valid_num1,
#                                 batch_valid_num2, batch_valid_max,
#                                 batch_valid_block_embedding1, batch_valid_block_embedding2,
#                                 batch_valid_inst_embedding1, batch_valid_inst_embedding2, batch_valid_inst_num1,
#                                 batch_valid_inst_num2])
#
#                 y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 , w2v_1, w2v_2,inst_max1,inst_max2,max_block\
#                     = get_batch(valid_label, valid_cfg_1, valid_cfg_2, valid_func_fea_1, valid_func_fea_2,
#                                 valid_node_fea_1, valid_node_fea_2,
#                                 valid_num1, valid_num2, valid_max,
#                                 valid_block_embedding1, valid_block_embedding2,
#                                 valid_inst_embedding1, valid_inst_embedding2, valid_inst_num1, valid_inst_num2)
#                 predict = dis.eval(feed_dict={
#                 w2v_left: w2v_1, cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1,
#                 w2v_right: w2v_2, cdfg_right: cdfg_2, fea_right: fea_2,v_num_right: v_num_2,
#                 max_num:max_block,
#                 labels: y, dropout_f: 0.9})
#                 tr_acc = compute_accuracy(predict, y)
#                 valid_loss = loss.eval(feed_dict={labels: y, dis: predict})
#                 avg_loss += valid_loss
#                 avg_acc += tr_acc * 100
#                 if is_debug:
#                     print ('valid     tr_acc %0.2f    tr_loss %0.2f  '%(tr_acc,valid_loss))
#             duration = time.time() - valid_start_time
#             # print 'valid set, %d,  time, %f, loss, %0.5f, acc, %0.2f' % (
#             #    iter, duration, avg_loss / (int(valid_num / B)), avg_acc / (int(valid_num / B)))
#             # saver.save(sess,
#             #            config.MODEL_VULSEEKER_DIR + os.sep + "vulseeker-model" + PREFIX + "_" + str(iter) + "bert_only.ckpt")
#             if iter == max_iter:
#                 fpr_file = config.FIT_RESULT_DIR + os.sep + PREFIX + '_' + str(iter) + '_fpr.csv'
#                 tpr_file = config.FIT_RESULT_DIR + os.sep + PREFIX + '_' + str(iter) + '_tpr.csv'
#             total_labels = []
#             total_predicts = []
#             avg_loss = 0.
#             avg_acc = 0.
#             test_total_batch = int(test_num / B)
#             start_time = time.time()
#             # Loop over all batches
#             # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_num
#             print (test_total_batch)
#             for m in range(test_total_batch):  # 原数据：test_total_batch
#                 test_label, test_cfg_1, test_cfg_2, test_func_fea_1, test_func_fea_2, \
#                 test_node_fea_1, test_node_fea_2, test_num1, test_num2, test_max, \
#                 test_block_embedding1, test_block_embedding2, \
#                 test_inst_embedding1, test_inst_embedding2, test_inst_num1, test_inst_num2 \
#                     = sess.run(
#                     [batch_test_label, batch_test_cfg_1, batch_test_cfg_2, batch_test_dfg_1, batch_test_dfg_2,
#                      batch_test_fea_1, batch_test_fea_2, batch_test_num1, batch_test_num2, batch_test_max,
#                      batch_test_block_embedding1, batch_test_block_embedding2,
#                      batch_test_inst_embedding1, batch_test_inst_embedding2, batch_test_inst_num1,
#                      batch_test_inst_num2])
#
#                 y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 ,w2v_1, w2v_2,inst_max1,inst_max2,max_block \
#                     = get_batch(test_label, test_cfg_1, test_cfg_2, test_func_fea_1, test_func_fea_2,
#                                 test_node_fea_1, test_node_fea_2, test_num1, test_num2, test_max,
#                                 test_block_embedding1, test_block_embedding2,
#                                 test_inst_embedding1, test_inst_embedding2, test_inst_num1, test_inst_num2)
#
#                 predict = dis.eval(feed_dict={
#                 w2v_left: w2v_1, cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1,
#                 w2v_right: w2v_2, cdfg_right: cdfg_2, fea_right: fea_2,v_num_right: v_num_2,
#                 max_num:max_block,
#                 labels: y, dropout_f: 0.9})
#                 tr_acc = compute_accuracy(predict, y)
#                 test_loss = loss.eval(feed_dict={labels: y, dis: predict})
#                 avg_loss += test_loss
#                 avg_acc += tr_acc * 100
#                 total_labels.append(y)
#                 total_predicts.append(predict)
#                 if iter == max_iter:
#                     test_label_list.append(y)
#                     test_predict_list.append(predict)
#                 if is_debug:
#                     print ('test     %d    tr_acc %0.2f   tr_loss %0.2f' % (m, tr_acc,test_loss))
#             duration = time.time() - start_time
#             total_labels = np.reshape(total_labels, (-1))
#             total_predicts = np.reshape(total_predicts, (-1))
#             fpr, tpr, AUC, precision, recall, f1score = calculate_auc(total_labels, total_predicts)
#             print (AUC)
#             print ('test set, time, %f, loss, %0.5f, acc, %0.2f' % (
#             duration, avg_loss / test_total_batch, avg_acc / test_total_batch))
#             test_loss_list.append(avg_loss/ test_total_batch)
#             test_acc_list.append(avg_acc/ test_total_batch)
#
#             AUC_list.append(AUC)
#             precision_list.append(precision)
#             recall_list.append(recall)
#             accuracy_list.append((avg_acc / test_total_batch))
#             f1score_list.append(f1score)
#
#             if iter == max_iter:
#                 write_results(fpr_file, fpr)
#                 write_results(tpr_file, tpr)
#
#
#     print ("end")
#
#     write_results(accuracy_file, accuracy_list)
#     write_results(precision_file, precision_list)
#     write_results(AUC_file, AUC_list)
#     write_results(f1score_file, f1score_list)
#     write_results(recall_file, recall_list)
#
#     com_label_predict(train_label_list,train_predict_list,train_label_predict_file)
#     com_label_predict(test_label_list,test_predict_list,test_label_predict_file)
#     plot_acc_loss('test',test_loss_list,test_acc_list)
#
#     # 保存模型
#     # saver.save(sess, config.MODEL_VULSEEKER_DIR + os.sep + "vulseeker-model" + PREFIX + "_final_bert_only.ckpt")
#
#     # coord.request_stop()
#     # coord.join(threads)

#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import sys
sys.path.append('..')
import config_jxd as config
# import train_safe_gpu
# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import host_subplot



# ===========  global parameters  ===========

T = 5  # iteration
N = 2  # embedding_depth
P = 64  # embedding_size
D = 11  # dimensional,feature num
B = 10  # mini-batch
lr = 0.0001  # learning_rate
# MAX_SIZE = 0 # record the max number of a function's block
max_iter = 10  # 原数据：100
decay_steps = 10  # 衰减步长
decay_rate = 0.0001  # 衰减率
snapshot = 1
is_debug = True
n_layer = 3
block_embedding = 768
BE = 768
BED = 768
IE = 100

# program_name = "coreutils"
# program_version = "6.12"
# program_arch = "arm-mips-x86"
# program_opti = "O1"

# program_name = "coreutils"
# program_version = "6.12"
# program_arch = "arm"
# program_opti = "O0-O1-O2-O3"

# program_name = "openssl"
# program_version = "1.0.1u"
# program_arch = "arm"
# program_opti = "O2-O3"

program_name = "busybox"
program_version = "1.27.2"
program_arch = "arm"
program_opti = "O0-O3"


program_info = (program_name, program_version, program_arch, program_opti)

PREFIX = program_name + '_' + program_version + '_' + program_arch + '_' + program_opti
TRAIN_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "train_" + PREFIX + ".tfrecord"
TEST_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "test_" + PREFIX + ".tfrecord"
VALID_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "valid_" + PREFIX + ".tfrecord"

precision_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_precision.csv"
AUC_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_AUC.csv"
recall_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_recall.csv"
f1score_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_f1score.csv"
accuracy_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_accuarcy.csv"

train_label_predict_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_train_label_predict_r.csv"
test_label_predict_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_test_label_predict_r.csv"

csv_file = config.DATASET_DIR
# =============== convert the real data to training data ==============
#       1.  construct_learning_dataset() combine the dataset list & real data
#       1-1. generate_adj_matrix_pairs()    traversal list and construct all the matrixs
#       1-1-1. convert_graph_to_adj_matrix()    process each cfg
#       1-2. generate_features_pair() traversal list and construct all functions' feature map
# =====================================================================
""" Parameter P = 64, D = 8, T = 7, N = 2,                  B = 10
     X_v = D * 1   <--->   8 * v_num * 10
     W_1 = P * D   <--->   64* 8    W_1 * X_v = 64*1
    mu_0 = P * 1   <--->   64* 1
     P_1 = P * P   <--->   64*64
     P_2 = P * P   <--->   64*64
    mu_2/3/4/5 = P * P     <--->  64*1
    W_2 = P * P     <--->  64*64
"""


def gcn_layer(cdfg, X, W, name='gcn_layer'):
    with tf.variable_scope(name):
        # testc = tf.einsum('abc,acd->abd', cdfg,X)
        message = tf.einsum('abc,cd->abd', X, W)
        # message1 = tf.einsum('abc,cd->abd', message, W)
        # message2 = tf.einsum('abc,cd->abd', message1, W)

        output = tf.einsum('abc,acd->abd', cdfg, message)
        # print 123

    return tf.nn.relu(output)


def self_gating(embeddings, weights_gate, weights_gatebias):
    gating = tf.nn.sigmoid(tf.matmul(embeddings, weights_gate) + weights_gatebias)

    return tf.multiply(embeddings, gating)



def structure2vec_net(cdfgs, x, v_num):
    """
    iput: [B,None,None] [B,None,D]

    """
    with tf.variable_scope("structure2vec_net") as structure2vec_net:
        B_mu_5 = tf.Variable(tf.zeros(shape=[0, P]), trainable=False)
        weights = {}
        # w_1 = tf.get_variable('w_1', [D + BE + IE, P], tf.float32,
        #                       tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        # w_1 = tf.get_variable('w_1', [D + BE , P], tf.float32,
        w_1 = tf.get_variable('w_1', [BED, P], tf.float32,
                              tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        w_2 = tf.get_variable('w_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        for i in range(n_layer):
            weights['w1%d' % (i + 1)] = tf.get_variable('w1%d' % (i + 1), [P, P], tf.float32,
                                                        tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            weights['w2%d' % (i + 1)] = tf.get_variable('w2%d' % (i + 1), [P, P], tf.float32,
                                                        tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # input [B, max_node, D]
        node_embeddings = tf.einsum('abc,cd->abd', x, w_1)  # [B, max_node, P]
        cur_node_embeddings = tf.nn.relu(node_embeddings)
        # propagation layer
        # cdfgs [B, max_node,max_node]
        for i in range(n_layer):
            cur_node_embeddings = tf.matmul(cdfgs, cur_node_embeddings)  # message aggregation
            cur_node_embeddings = tf.einsum('abc,cd->abd', cur_node_embeddings, weights['w1%d' % (i + 1)])
            if i + 1 < n_layer:
                cur_node_embeddings = tf.nn.relu(cur_node_embeddings)
            tot_node_embeddings = cur_node_embeddings + node_embeddings
            cur_node_embeddings = tf.nn.tanh(tot_node_embeddings)

        g_embed = tf.reduce_sum(cur_node_embeddings, 1)  # [batch, P]
        output = tf.matmul(g_embed, w_2)

        return output



def calculate_auc(labels, predicts):
    fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=1)

    new_predicts = []

    for i in range(len(predicts)):
        if labels[i] == 1:
            if predicts[i] > 0.8:
                new_predicts.append(labels[i])
            else:
                new_predicts.append(-1)

        if labels[i] == -1:
            if predicts[i] < 0.8:
                new_predicts.append(labels[i])
            else:
                new_predicts.append(1)

    precision = precision_score(labels, new_predicts, pos_label=1)
    recall = recall_score(labels, new_predicts)
    f1 = f1_score(labels, new_predicts)

    AUC = auc(fpr, tpr)
    print ("auc : ", AUC)
    return fpr, tpr, AUC, precision, recall, f1


def contrastive_loss(labels, distance):
    #    tmp= y * tf.square(d)
    #    #tmp= tf.mul(y,tf.square(d))
    #    tmp2 = (1-y) * tf.square(tf.maximum((1 - d),0))
    #    return tf.reduce_sum(tmp +tmp2)/B/2
    #    print "contrastive_loss", y,
    loss = tf.to_float(tf.reduce_sum(tf.square(distance - labels)))
    return loss


def tf_cosine_distance(tensor1, tensor2):
    tensor1 = tf.reshape(tensor1, shape=(1, -1))
    tensor2 = tf.reshape(tensor2, shape=(1, -1))

    # 求模长
    tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
    tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))

    # 内积
    tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
    cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)

    return cosin


def generate_simmat(ori_vector, aug_vector, temp):
    for i in range(5):
        sim = tf.expand_dims(tf_cosine_distance(ori_vector[i], aug_vector) * 1 / temp, 0)

        if i == 0:
            sim_score = sim
        else:
            sim_score = tf.concat([sim_score, sim], axis=0)

    return sim_score


def newloss(model1, model2):
    for i in range(5):
        qe = tf.expand_dims(generate_simmat(model2, model1[i], 0.8), 0)

        if i == 0:
            mo = qe
        else:
            mo = tf.concat([mo, qe], 0)

    for i in range(5):
        molecular = tf.math.exp(mo[i][i])
        # denominator = tf.math.exp(tf.reduce_sum(mo[i],0))
        denominator = tf.reduce_sum(tf.math.exp(mo[i]), 0)
        if i == 0:
            result1 = tf.log(molecular / denominator)
        else:
            result1 = result1 + tf.log(molecular / denominator)

    mo2 = tf.transpose(mo)
    for i in range(5):
        molecular = tf.math.exp(mo2[i][i])
        # denominator = tf.math.exp(tf.reduce_sum(mo[i], 0) - mo[i][i])
        denominator = tf.reduce_sum(tf.math.exp(mo2[i]), 0)
        # denominator = tf.reduce_sum(tf.math.exp(mo[i]), 0)
        if i == 0:
            result2 = tf.log(molecular / denominator)
        else:
            result2 = result2 + tf.log(molecular / denominator)

    return -result1 - result2


def compute_accuracy(prediction, labels):
    accu = 0.0
    threshold = 0.8
    for i in range(len(prediction)):
        if labels[i][0] == 1:
            if prediction[i][0] > threshold:
                accu += 1.0
        else:
            if prediction[i][0] < threshold:
                accu += 1.0
    acc = accu / len(prediction)
    return acc


def cal_distance(model1, model2):
    a_b = tf.reduce_sum(tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(model1, (1, -1)),
                                                             tf.reshape(model2, (1, -1))], 0), 0), (B, P)), 1,
                        keep_dims=True)
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(model1), 1, keep_dims=True))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(model2), 1, keep_dims=True))
    distance = a_b / tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(a_norm, (1, -1)),
                                                          tf.reshape(b_norm, (1, -1))], 0), 0), (B, 1))
    return distance


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example

    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'cfg_1': tf.FixedLenFeature([], tf.string),
        'cfg_2': tf.FixedLenFeature([], tf.string),
        'func_fea_1': tf.FixedLenFeature([], tf.string),
        'func_fea_2': tf.FixedLenFeature([], tf.string),
        'node_fea_1': tf.FixedLenFeature([], tf.string),
        'node_fea_2': tf.FixedLenFeature([], tf.string),
        'num1': tf.FixedLenFeature([], tf.int64),
        'num2': tf.FixedLenFeature([], tf.int64),
        'max': tf.FixedLenFeature([], tf.int64),
        'inst_embedding1': tf.FixedLenFeature([], tf.string),
        'inst_embedding2': tf.FixedLenFeature([], tf.string),
        'inst_num1': tf.FixedLenFeature([], tf.string),
        'inst_num2': tf.FixedLenFeature([], tf.string)
    })

    label = tf.cast(features['label'], tf.int32)

    cfg_1 = features['cfg_1']
    cfg_2 = features['cfg_2']

    func_fea_1 = features['func_fea_1']
    func_fea_2 = features['func_fea_2']

    num1 = tf.cast(features['num1'], tf.int32)
    node_fea_1 = features['node_fea_1']

    num2 = tf.cast(features['num2'], tf.int32)
    node_fea_2 = features['node_fea_2']

    max_num = tf.cast(features['max'], tf.int32)

    block_embedding1 = features['inst_embedding1']
    block_embedding2 = features['inst_embedding2']

    return label, cfg_1, cfg_2, func_fea_1, func_fea_2, node_fea_1, node_fea_2, num1, num2, max_num, block_embedding1, block_embedding2


def create_norm_adjaceny(adj):
    # pre adjcency matrix
    new_adj = adj + np.identity(adj.shape[0])
    # rowsum = np.array(adj.sum(1))
    # d_inv = np.power(rowsum, -0.5).flatten()
    # d_inv[np.isinf(d_inv)] = 0.
    # d_mat_inv = np.diag(d_inv)
    # norm_adj_tmp = d_mat_inv.dot(new_adj)
    # adj_matrix = norm_adj_tmp.dot(d_mat_inv)
    # return adj_matrix
    return new_adj


def write_results(csv_file, result_list):
    with open(csv_file, "w") as fp:
        for e in result_list:
            res_str = str(e) + '\n'
            fp.write(res_str)

    fp.close()


def get_datasetnum(csv_file, program_info):

    test_dataset_num = 0
    train_dataset_num = 0
    valid_dataset_num = 0

    for csv_name in os.listdir(csv_file):
        csv_name_list = csv_name.split('_')
        csv_info = (csv_name_list[1], csv_name_list[2], csv_name_list[3], csv_name_list[4])

        if csv_name_list[0] == "test" and csv_info == program_info:
            with open(csv_file + os.sep + csv_name, 'r') as f:
                test_dataset_num = len(f.readlines())

        if csv_name_list[0] == "train" and csv_info == program_info:
            with open(csv_file + os.sep + csv_name, 'r') as f:
                train_dataset_num = len(f.readlines())

        if csv_name_list[0] == "valid" and csv_info == program_info:
            with open(csv_file + os.sep + csv_name, 'r') as f:
                valid_dataset_num = len(f.readlines())

    return train_dataset_num, valid_dataset_num, test_dataset_num


def com_label_predict(labels, predicts, lables_predicts_file):
    labels_r = np.reshape(labels, (-1))
    predicts_r = np.reshape(predicts, (-1))

    with open(lables_predicts_file, "w") as lpf:
        for i in range(len(predicts_r)):
            label = labels_r[i]

            predict = predicts_r[i]

            res_str = str(predict) + ',' + str(label) + '\n'

            lpf.write(res_str)

    lpf.close()


def get_batch(label, cfg_str1, cfg_str2, dfg_str1, dfg_str2, fea_str1, fea_str2, num1, num2, max_num, blockembed1,
              blockembed2):
    y = np.reshape(label, [B, 1])

    v_num_1 = []
    v_num_2 = []
    for i in range(B):
        v_num_1.append([int(num1[i])])
        v_num_2.append([int(num2[i])])

    # 补齐 martix 矩阵的长度
    # cdfg_1 = []
    # cdfg_2 = []
    cfg_1 = []
    cfg_2 = []
    for i in range(B):
        cfg_arr = np.array(cfg_str1[i].decode().split(','))
        cfg_adj = np.reshape(cfg_arr, (int(num1[i]), int(num1[i])))  # reshape成邻接矩阵
        cfg_ori1 = cfg_adj.astype(np.float32)
        cfg_ori1 = create_norm_adjaceny(cfg_ori1)
        cfg_zero1 = np.zeros([int(num1[i]), (int(max_num[i]) - int(num1[i]))])
        cfg_zero2 = np.zeros([(int(max_num[i]) - int(num1[i])), int(max_num[i])])
        cfg_vec1 = np.concatenate([cfg_ori1, cfg_zero1], axis=1)
        cfg_vec2 = np.concatenate([cfg_vec1, cfg_zero2], axis=0)
        cfg_1.append(cfg_vec2.tolist())

        cfg_arr = np.array(cfg_str2[i].decode().split(','))
        cfg_adj = np.reshape(cfg_arr, (int(num2[i]), int(num2[i])))
        cfg_ori2 = cfg_adj.astype(np.float32)
        cfg_ori2 = create_norm_adjaceny(cfg_ori2)
        cfg_zero1 = np.zeros([int(num2[i]), (int(max_num[i]) - int(num2[i]))])
        cfg_zero2 = np.zeros([(int(max_num[i]) - int(num2[i])), int(max_num[i])])
        cfg_vec1 = np.concatenate([cfg_ori2, cfg_zero1], axis=1)
        cfg_vec2 = np.concatenate([cfg_vec1, cfg_zero2], axis=0)
        cfg_2.append(cfg_vec2.tolist())

    # 补齐 feature 列表的长度
    fea_1 = []
    fea_2 = []
    for i in range(B):
        block_embed_arr1 = np.array(blockembed1[i].decode().split(','))
        block_embed_ori1 = block_embed_arr1.astype(np.float32)
        block_embed_vec1 = np.reshape(block_embed_ori1, [int(num1[i]), BE])

        block_zero1 = np.zeros([int(max_num[i] - num1[i]), BE])
        block_vec1 = np.concatenate([block_embed_vec1, block_zero1], axis=0)

        # fea_arr = np.array(fea_str1[i].decode().split(','))
        # fea_ori = fea_arr.astype(np.float32)
        # fea_ori1 = np.resize(fea_ori, (int(num1[i]), D))
        # fea_zero1 = np.zeros([int(max_num[i] - num1[i]), D])
        # fea_vec1 = np.concatenate([fea_ori1, fea_zero1], axis=0)
        #
        # fea_vec1 = np.concatenate([fea_vec1, block_vec1], axis=1)

        # fea_1.append(fea_vec1)
        fea_1.append(block_vec1)

        block_embed_arr2 = np.array(blockembed2[i].decode().split(','))
        block_embed_ori2 = block_embed_arr2.astype(np.float32)
        block_embed_vec2 = np.reshape(block_embed_ori2, [int(num2[i]), BE])

        block_zero2 = np.zeros([int(max_num[i] - num2[i]), BE])
        block_vec2 = np.concatenate([block_embed_vec2, block_zero2], axis=0)

        # fea_arr = np.array(fea_str2[i].decode().split(','))
        # fea_ori = fea_arr.astype(np.float32)
        # fea_ori2 = np.resize(fea_ori, (int(num2[i]), D))
        # fea_zero2 = np.zeros([int(max_num[i] - num2[i]), D])
        # fea_vec2 = np.concatenate([fea_ori2, fea_zero2], axis=0)
        #
        # fea_vec2 = np.concatenate([fea_vec2, block_vec2], axis=1)
        # fea_2.append(fea_vec2)
        fea_2.append(block_vec2)

    return y, cfg_1, cfg_2, fea_1, fea_2, v_num_1, v_num_2


# 4.construct the network
# Initializing the variables
# Siamese network major part

# Initializing the variables

init = tf.global_variables_initializer()
# init  = tf.compat.v1.global_variables_initializer()
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase=True)

v_num_left = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_left')
cdfg_left = tf.placeholder(tf.float32, shape=([B, None, None]), name='cdfg_left')
fea_left = tf.placeholder(tf.float32, shape=([B, None, BED]), name='fea_left')

v_num_right = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_right')
cdfg_right = tf.placeholder(tf.float32, shape=([B, None, None]), name='cdfg_right')
fea_right = tf.placeholder(tf.float32, shape=([B, None, BED]), name='fea_right')

labels = tf.placeholder(tf.float32, shape=([B, 1]), name='gt')

dropout_f = tf.placeholder("float")

with tf.variable_scope("siamese") as siamese:
    model1 = structure2vec_net(cdfg_left, fea_left, v_num_left)
    siamese.reuse_variables()
    model2 = structure2vec_net(cdfg_right, fea_right, v_num_right)

dis = cal_distance(model1, model2)

loss = contrastive_loss(labels, dis)

newloss = newloss(model1, model2)

# def reg_loss():
# reg_loss() = 0
# for key in weights:
# reg_loss += 0.001*tf.nn.l2_loss(weights[key])
# return reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(newloss)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, epsilon=1e-10).minimize(loss)


# 三大数据：基本块特征，控制流图，数据流图
# list_train_label：
# cfg :控制流图 dfg:数据流图  fea:基本块特征

list_train_label, list_train_cfg_1, list_train_cfg_2, list_train_func_fea_1, list_train_func_fea_2, list_train_node_fea_1, \
    list_train_node_fea_2, list_train_num1, list_train_num2, list_train_max, \
    list_train_block_embedding1, list_train_block_embedding2, \
    = read_and_decode(TRAIN_TFRECORD)

batch_train_label, batch_train_cfg_1, batch_train_cfg_2, batch_train_func_fea_1, batch_train_func_fea_2, batch_train_node_fea_1, \
    batch_train_node_fea_2, batch_train_num1, batch_train_num2, batch_train_max, \
    batch_train_block_embedding1, batch_train_block_embedding2, \
    = tf.train.batch(
    [list_train_label, list_train_cfg_1, list_train_cfg_2, list_train_func_fea_1, list_train_func_fea_2,
     list_train_node_fea_1, list_train_node_fea_2, list_train_num1, list_train_num2, list_train_max,
     list_train_block_embedding1, list_train_block_embedding2],
    batch_size=B, capacity=10)

list_valid_label, list_valid_cfg_1, list_valid_cfg_2, list_valid_dfg_1, list_valid_dfg_2, list_valid_fea_1, \
    list_valid_fea_2, list_valid_num1, list_valid_num2, list_valid_max, \
    list_valid_block_embedding1, list_valid_block_embedding2, \
    = read_and_decode(VALID_TFRECORD)

batch_valid_label, batch_valid_cfg_1, batch_valid_cfg_2, batch_valid_dfg_1, batch_valid_dfg_2, batch_valid_fea_1, \
    batch_valid_fea_2, batch_valid_num1, batch_valid_num2, batch_valid_max, \
    batch_valid_block_embedding1, batch_valid_block_embedding2, \
    = tf.train.batch([list_valid_label, list_valid_cfg_1, list_valid_cfg_2, list_valid_dfg_1, list_valid_dfg_2,
                      list_valid_fea_1, list_valid_fea_2, list_valid_num1, list_valid_num2, list_valid_max,
                      list_valid_block_embedding1, list_valid_block_embedding2],
                     batch_size=B, capacity=10)

list_test_label, list_test_cfg_1, list_test_cfg_2, list_test_dfg_1, list_test_dfg_2, list_test_fea_1, \
    list_test_fea_2, list_test_num1, list_test_num2, list_test_max, \
    list_test_block_embedding1, list_test_block_embedding2, \
    = read_and_decode(TEST_TFRECORD)

batch_test_label, batch_test_cfg_1, batch_test_cfg_2, batch_test_dfg_1, batch_test_dfg_2, batch_test_fea_1, \
    batch_test_fea_2, batch_test_num1, batch_test_num2, batch_test_max, \
    batch_test_block_embedding1, batch_test_block_embedding2, \
    = tf.train.batch([list_test_label, list_test_cfg_1, list_test_cfg_2, list_test_dfg_1, list_test_dfg_2,
                      list_test_fea_1, list_test_fea_2, list_test_num1, list_test_num2, list_test_max,
                      list_test_block_embedding1, list_test_block_embedding2],
                     batch_size=B, capacity=10)

init_opt = tf.global_variables_initializer()
saver = tf.train.Saver()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True

with tf.Session(config=tfconfig) as sess:
    # writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init_opt)

    # if config.SETP5_IF_RESTORE_VULSEEKER_MODEL:
    #     saver.restore(sess, config.MODEL_VULSEEKER_DIR + os.sep + config.STEP5_VULSEEKER_MODEL_TO_RESTORE)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Training cycle
    iter = 0

    train_num, valid_num, test_num = get_datasetnum(csv_file, program_info)
    print(train_num, valid_num, test_num)

    precision_list = []
    AUC_list = []
    recall_list = []
    accuracy_list = []
    f1score_list = []

    train_label_list = []
    train_predict_list = []
    test_label_list = []
    test_predict_list = []

    while iter < max_iter:
        iter += 1
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(train_num / B)
        start_time = time.time()
        # Loop over all batches
        # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_nu
        print (total_batch)

        for i in range(total_batch):  # 原数据：total_batch
            train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1, train_node_fea_2, \
                train_num1, train_num2, train_max, \
                train_block_embedding1, train_block_embedding2, \
                = sess.run([batch_train_label, batch_train_cfg_1, batch_train_cfg_2, batch_train_func_fea_1,
                            batch_train_func_fea_2, batch_train_node_fea_1, batch_train_node_fea_2, batch_train_num1,
                            batch_train_num2, batch_train_max,
                            batch_train_block_embedding1, batch_train_block_embedding2])

            y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 \
                = get_batch(train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1,
                            train_node_fea_2,
                            train_num1, train_num2, train_max,
                            train_block_embedding1, train_block_embedding2)

            _, loss_value, predict = sess.run([optimizer, loss, dis], feed_dict={
                cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1, cdfg_right: cdfg_2, fea_right: fea_2,
                v_num_right: v_num_2, labels: y, dropout_f: 0.9})

            if iter == 10:
                train_label_list.append(y)
                train_predict_list.append(predict)

            tr_acc = compute_accuracy(predict, y)
            if is_debug :
                print ('     %d   tr_acc %0.2f    tr_loss  %0.2f' % (i, tr_acc, loss_value))
                # sys.stdout.flush()
            avg_loss += loss_value
            avg_acc += tr_acc * 100
        duration = time.time() - start_time

        if iter % snapshot == 0:
            # validing model
            avg_loss = 0.
            avg_acc = 0.
            valid_start_time = time.time()
            print (int(valid_num / B))
            for m in range(int(valid_num / B)):  # 原数据：int(valid_num / B)
                valid_label, valid_cfg_1, valid_cfg_2, valid_func_fea_1, valid_func_fea_2, valid_node_fea_1, valid_node_fea_2, \
                    valid_num1, valid_num2, valid_max, \
                    valid_block_embedding1, valid_block_embedding2, \
                    = sess.run([batch_valid_label, batch_valid_cfg_1, batch_valid_cfg_2, batch_valid_dfg_1,
                                batch_valid_dfg_2, batch_valid_fea_1, batch_valid_fea_2, batch_valid_num1,
                                batch_valid_num2, batch_valid_max,
                                batch_valid_block_embedding1, batch_valid_block_embedding2])

                y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 \
                    = get_batch(valid_label, valid_cfg_1, valid_cfg_2, valid_func_fea_1, valid_func_fea_2,
                                valid_node_fea_1, valid_node_fea_2,
                                valid_num1, valid_num2, valid_max,
                                valid_block_embedding1, valid_block_embedding2)
                predict = dis.eval(feed_dict={
                    cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1, cdfg_right: cdfg_2,
                    fea_right: fea_2, v_num_right: v_num_2, labels: y, dropout_f: 0.9})
                tr_acc = compute_accuracy(predict, y)
                avg_loss += loss.eval(feed_dict={labels: y, dis: predict})
                avg_acc += tr_acc * 100
                if is_debug:
                    print ('     tr_acc %0.2f    tr_loss %0.2f  ' % (tr_acc, loss_value))
            duration = time.time() - valid_start_time
            # print ('valid set, %d,  time, %f, loss, %0.5f, acc, %0.2f' % (
            #     iter, duration, avg_loss / (int(valid_num / B)), avg_acc / (int(valid_num / B))))
            # saver.save(sess, config.MODEL_VULSEEKER_DIR + os.sep + "vulseeker-model"+PREFIX+"_"+str(iter)+".ckpt")

            if iter == 10:
                fpr_file = config.BERT_RESULT_DIR + os.sep + PREFIX + '_' + str(iter) + '_fpr.csv'
                tpr_file = config.BERT_RESULT_DIR + os.sep + PREFIX + '_' + str(iter) + '_tpr.csv'

            total_labels = []
            total_predicts = []
            avg_loss = 0.
            avg_acc = 0.
            test_total_batch = int(test_num / B)
            start_time = time.time()
            # Loop over all batches
            # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_num
            print (test_total_batch)
            for m in range(test_total_batch):  # 原数据：test_total_batch
                test_label, test_cfg_1, test_cfg_2, test_func_fea_1, test_func_fea_2, \
                    test_node_fea_1, test_node_fea_2, test_num1, test_num2, test_max, \
                    test_block_embedding1, test_block_embedding2, \
                    = sess.run(
                    [batch_test_label, batch_test_cfg_1, batch_test_cfg_2, batch_test_dfg_1, batch_test_dfg_2,
                     batch_test_fea_1, batch_test_fea_2, batch_test_num1, batch_test_num2, batch_test_max,
                     batch_test_block_embedding1, batch_test_block_embedding2])

                y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 \
                    = get_batch(test_label, test_cfg_1, test_cfg_2, test_func_fea_1, test_func_fea_2,
                                test_node_fea_1, test_node_fea_2, test_num1, test_num2, test_max,
                                test_block_embedding1, test_block_embedding2)
                predict = dis.eval(
                    feed_dict={cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1, cdfg_right: cdfg_2,
                               fea_right: fea_2, v_num_right: v_num_2, labels: y, dropout_f: 1.0})
                tr_acc = compute_accuracy(predict, y)
                avg_loss += loss.eval(feed_dict={labels: y, dis: predict})
                avg_acc += tr_acc * 100
                total_labels.append(y)
                total_predicts.append(predict)

                if iter == 10:
                    test_label_list.append(y)
                    test_predict_list.append(predict)

                if is_debug and m % 10 == 0:
                    print('     %d    tr_acc %0.2f   tr_loss %0.2f' % (m, tr_acc, loss_value))
            duration = time.time() - start_time
            total_labels = np.reshape(total_labels, (-1))
            total_predicts = np.reshape(total_predicts, (-1))

            fpr, tpr, AUC, precision, recall, f1score = calculate_auc(total_labels, total_predicts)
            print (AUC)
            print ('test set, time, %f, loss, %0.5f, acc, %0.2f' % ( duration, avg_loss / test_total_batch, avg_acc / test_total_batch))

            AUC_list.append(AUC)
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append((avg_acc / test_total_batch))
            f1score_list.append(f1score)

            if iter == 10:
                write_results(fpr_file, fpr)
                write_results(tpr_file, tpr)

    print ("end")

    write_results(accuracy_file, accuracy_list)
    write_results(precision_file, precision_list)
    write_results(AUC_file, AUC_list)
    write_results(f1score_file, f1score_list)
    write_results(recall_file, recall_list)

    com_label_predict(train_label_list, train_predict_list, train_label_predict_file)
    com_label_predict(test_label_list, test_predict_list, test_label_predict_file)













# #!/usr/bin/env python
# # _*_ coding: utf-8 _*_
#
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import numpy as np
# import time
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import precision_score, recall_score, f1_score
# import os
# import sys
# sys.path.append('..')
# import config_jxd as config
# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import host_subplot
#
#
#
# # ===========  global parameters  ===========
#
# T = 5  # iteration
# N = 2  # embedding_depth
# P = 64  # embedding_size
# D = 8  # dimensional,feature num
# B = 10  # mini-batch
# lr = 0.0001  # learning_rate
# # MAX_SIZE = 0 # record the max number of a function's block
# max_iter = 10  # 原数据：100
# decay_steps = 10  # 衰减步长
# decay_rate = 0.0001  # 衰减率
# snapshot = 1
# is_debug = True
# n_layer = 3
# block_embedding = 768
# BE = 768
# BED = 776
# IE = 100
# MAX_I = 20
#
#
#
# # train_num = 5000
# # valid_num = int(train_num / 10)
# # test_num = int(train_num / 10)
# # PREFIX = "_coreutils_1000"
# # TRAIN_TFRECORD="TFrecord/train_vulSeeker_data"+PREFIX+".tfrecord"
# # TEST_TFRECORD="TFrecord/test_vulSeeker_data"+PREFIX+".tfrecord"
# # VALID_TFRECORD="TFrecord/valid_vulSeeker_data"+PREFIX+".tfrecord"
# # PREFIX = "_" + str(config.TRAIN_DATASET_NUM) + "_[" + '_'.join(config.STEP3_PORGRAM_ARR) + "]"
# # TRAIN_TFRECORD = '/home/ubuntu/Desktop/eee/4_EmbeddingTFRecord/zippp' + os.sep + "train_"+"openssl_1.0.1f_w2v_bert"
# # TEST_TFRECORD = '/home/ubuntu/Desktop/eee/4_EmbeddingTFRecord/zippp' + os.sep + "train_"+"openssl_1.0.1f_w2v_bert"
# # VALID_TFRECORD = '/home/ubuntu/Desktop/eee/4_EmbeddingTFRecord/zippp' + os.sep + "valid_" + "openssl_1.0.1f_w2v_bert"
#
# # bert_TRAIN_TFRECORD = './bert_tf_record' + os.sep + "train_"+"single_arch_ft_pad512_last4layer_means.tfrecord"
# # bert_TEST_TFRECORD = './bert_tf_record' + os.sep + "test_"+"single_arch_ft_pad512_last4layer_means.tfrecord"
# # bert_VALID_TFRECORD = './bert_tf_record' + os.sep + "valid_"+"single_arch_ft_pad512_last4layer_means.tfrecord"
#
#
#
# # program_name = "coreutils"
# # program_version = "6.12"
# # program_arch = "arm"
# # program_opti = "O0-O1-O2-O3"
#
# program_name = "openssl"
# program_version = "1.0.1f"
# program_arch = "arm-mips-x86"
# program_opti = "O1"
#
# # program_name = "busybox"
# # program_version = "1.27.2"
# # program_arch = "arm"
# # program_opti = "O0-O3"
#
# PREFIX = program_name + '_' + program_version + '_' + program_arch + '_' + program_opti
# # TRAIN_TFRECORD = '/home/ubuntu/Desktop/eee/4_EmbeddingTFRecord/zippp/train_openssl_1.0.1f_w2v_bert'
# # TEST_TFRECORD = '/home/ubuntu/Desktop/eee/4_EmbeddingTFRecord/zippp/test_openssl_1.0.1f_w2v_bert'
# # VALID_TFRECORD = '/home/ubuntu/Desktop/eee/4_EmbeddingTFRecord/zippp/valid_openssl_1.0.1f_w2v_bert'
# TRAIN_TFRECORD = config.TFRECORD_EMBEDDING_DIR_OLD + os.sep + "train_" + PREFIX + ".tfrecord"
# TEST_TFRECORD = config.TFRECORD_EMBEDDING_DIR_OLD + os.sep + "test_" + PREFIX + ".tfrecord"
# VALID_TFRECORD = config.TFRECORD_EMBEDDING_DIR_OLD + os.sep + "valid_" + PREFIX + ".tfrecord"
# program_info = (program_name, program_version, program_arch, program_opti)
#
#
# precision_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_precision.csv"
# AUC_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_AUC.csv"
# recall_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_recall.csv"
# f1score_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_f1score.csv"
# accuracy_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_accuarcy.csv"
#
#
# train_label_predict_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_train_label_predict_r.csv"
# test_label_predict_file = config.BERT_RESULT_DIR + os.sep + PREFIX + "_test_label_predict_r.csv"
#
#
# csv_file = config.DATASET_DIR
# # result_file = config.ZTHtest + os.sep + "test" + str(config.TRAIN_DATASET_NUM) + "_[" + '_'.join(
# #     config.STEP3_PORGRAM_ARR) + "].txt"
#
# # result_fp = open(result_file, "w")
# # =============== convert the real data to training data ==============
# #       1.  construct_learning_dataset() combine the dataset list & real data
# #       1-1. generate_adj_matrix_pairs()    traversal list and construct all the matrixs
# #       1-1-1. convert_graph_to_adj_matrix()    process each cfg
# #       1-2. generate_features_pair() traversal list and construct all functions' feature map
# # =====================================================================
#
# def block_embed(X_inst_embed,max_inst_num,inst_dim,n_hidden):
#
#
#     X = tf.reshape(X_inst_embed,[-1,max_inst_num,inst_dim])
#     X = tf.transpose(X,[1,0,2])
#     X = tf.reshape(X,[-1,inst_dim])
#     X = tf.split(X,max_inst_num,0)
#     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
#     outputs ,states = tf.nn.static_rnn(lstm_cell,X,dtype = tf.float32)
#     ret = tf.reshape(outputs[-1],[B,-1,n_hidden])
#     return ret
#
#
#
# def structure2vec_net(cdfgs, block_eb, inst_eb):
#     """
#     iput: [B,None,None] [B,None,D]
#
#     """
#     with tf.variable_scope("structure2vec_net") as structure2vec_net:
#         B_mu_5 = tf.Variable(tf.zeros(shape=[0, P]), trainable=False)
#         weights = {}
#         # w_1 = tf.get_variable('w_1', [D + BE + IE, P], tf.float32,
#         #                       tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         # w_1 = tf.get_variable('w_1', [D + BE , P], tf.float32,
#         # w_1 = tf.get_variable('w_1', [IE, P], tf.float32,
#         #                       tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         w_1 = tf.get_variable('w_1', [BE, P], tf.float32,
#                               tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         w_2 = tf.get_variable('w_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         for i in range(3):
#             weights['w1%d' % (i + 1)] = tf.get_variable('w1%d' % (i + 1), [P, P], tf.float32,
#                                                         tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#             weights['w2%d' % (i + 1)] = tf.get_variable('w2%d' % (i + 1), [P, P], tf.float32,
#                                                         tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         # w2v_lstm = block_embed(inst_eb,MAX_I,IE,IE)
#         # x = tf.concat([block_eb,w2v_lstm],axis=0)
#         x = block_eb
#         # x = w2v_lstm
#         node_embeddings = tf.einsum('abc,cd->abd', x, w_1)
#         # node_embeddings = tf.einsum('abc,cd->abd', w2v_lstm, w_1)# [B, max_node, P]
#         cur_node_embeddings = tf.nn.relu(node_embeddings)
#         # propagation layer
#         # cdfgs [B, max_node,max_node]
#         for i in range(n_layer):
#             cur_node_embeddings = tf.matmul(cdfgs, cur_node_embeddings)  # message aggregation
#             cur_node_embeddings = tf.einsum('abc,cd->abd', cur_node_embeddings, weights['w1%d' % (i + 1)])
#             if i + 1 < n_layer:
#                 cur_node_embeddings = tf.nn.relu(cur_node_embeddings)
#             tot_node_embeddings = cur_node_embeddings + node_embeddings
#             cur_node_embeddings = tf.nn.tanh(tot_node_embeddings)
#
#         g_embed = tf.reduce_sum(cur_node_embeddings, 1)  # [batch, P]
#         output = tf.matmul(g_embed, w_2)
#         return output
#
#
#
# def bi_rnn(X_inst_embed,max_inst_num,inst_dim):
#     X = tf.reshape(X_inst_embed,[-1,max_inst_num,inst_dim])
#     X = tf.transpose(X,[1,0,2])
#     X = tf.reshape(X,[-1,inst_dim])  #(batch*block_num*max_inst_num,inst_dim)
#     X = tf.split(X, max_inst_num, 0) # get a list of max_inst_num * (batch * block_num , inst_dim)
#     forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = inst_dim)
#     backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = inst_dim)
#     bi_rnn_output , _ , _ = tf.nn.static_bidirectional_rnn(forward_cell,backward_cell,X,dtype=tf.float32)
#     ret = tf.reshape(tf.stack(bi_rnn_output), [B,-1,max_inst_num,2*inst_dim])
#     return ret
#
#
# def safe_net(inst_eb):
#     with tf.variable_scope("safe_net") as safe_net:
#         r = 10
#         da = 100
#         e = 200
#         n = P
#         ws1 = tf.get_variable('ws1',[da,2*IE],tf.float32,tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         ws2 = tf.get_variable('ws2',[r,da],tf.float32,tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         H = bi_rnn(inst_eb,MAX_I,IE)#[Batch,block,MAX_I,2*IE] , [Batch,block,m,u]
#         HT = tf.transpose(H,[0,1,3,2]) #HT [Batch,block,2*IE,MAX_I]
#         w1_HT = tf.einsum('ec,abcd->abed',ws1,HT) #[batch,block,da,MAX_I],which is [batch,block,da,m]
#         tanh = tf.nn.tanh(w1_HT)
#         w2_tanh = tf.einsum('ec,abcd->abed',ws2,tanh)#[batch,block,r,m]
#         matrix_A = tf.nn.softmax(w2_tanh)
#         matrix_B = tf.einsum('abcd,abde->abce',matrix_A,H)#A*H->[batch,block,r,u],[batch,block,r,MAX_I]
#         flatten_B = tf.reshape(matrix_B,[B,-1,r * MAX_I])
#         reduce_sum_B = tf.reduce_sum(flatten_B, 1)  #[batch,r*u]
#         # weight_l1 = tf.get_variable('weight_l1',[r*MAX_I,r * MAX_I],tf.float32,
#         #                          tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
#         # weight_l2 = tf.get_variable('weight_l2', [r * MAX_I, r * MAX_I], tf.float32,
#         #                             tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#
#         layer0 = tf.layers.dense(reduce_sum_B,r*MAX_I)
#         # layer1 = tf.layers.dense(layer0,r*MAX_I)
#         out_W1 = tf.get_variable('out_W1',[e,r * MAX_I],tf.float32,
#                                  tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
#         out_W2 = tf.get_variable('out_W2', [n,e], tf.float32,
#                                  tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
#         mid1 = tf.einsum('ab,cb->ca',out_W1,layer0) #[batch , e]
#         mid2 = tf.nn.relu(mid1)
#         out = tf.einsum('ab,cb->ca',out_W2,mid2) # [batch,n]
#         return out
#
#
# # def calculate_auc(labels, predicts):
# #     fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=1)
# #     AUC = auc(fpr, tpr)
# #     print "auc : ", AUC
# #     return AUC
# def calculate_auc(labels, predicts):
#     fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=1)
#
#     new_predicts = []
#
#     for i in range(len(predicts)):
#         if labels[i] == 1:
#             if predicts[i] > 0.8:
#                 new_predicts.append(labels[i])
#             else:
#                 new_predicts.append(-1)
#
#         if labels[i] == -1:
#             if predicts[i] < 0.8:
#                 new_predicts.append(labels[i])
#             else:
#                 new_predicts.append(1)
#
#     precision = precision_score(labels, new_predicts, pos_label=1)
#     recall = recall_score(labels, new_predicts)
#     f1 = f1_score(labels, new_predicts)
#
#     AUC = auc(fpr, tpr)
#     print ("auc : ", AUC)
#     return fpr, tpr, AUC, precision, recall, f1
#
# def contrastive_loss(labels, distance):
#     #    tmp= y * tf.square(d)
#     #    #tmp= tf.mul(y,tf.square(d))
#     #    tmp2 = (1-y) * tf.square(tf.maximum((1 - d),0))
#     #    return tf.reduce_sum(tmp +tmp2)/B/2
#     #    print "contrastive_loss", y,
#     loss = tf.to_float(tf.reduce_sum(tf.square(distance - labels)))
#     return loss
#
#
# '''
# 张天豪修改　　　１１.４
# '''
#
#
# def get_positive_expectation(p_samples, average=True):
#
#     log_2 = math.log(2.)
#     Ep = log_2 - tf.nn.softplus(- p_samples)
#
#     return Ep
#
#
# '''
# 张天豪修改　　　１１．４
# '''
#
#
# def get_negative_expectation(q_samples, average=True):
#     """Computes the negative part of a JS Divergence.
#     Args:
#         q_samples: Negative samples.
#         average: Average the result over samples.
#     Returns:
#         th.Tensor
#     """
#     log_2 = math.log(2.)
#     Eq = tf.nn.softplus(-q_samples) + q_samples - log_2
#
#     return Eq
#
#
# def cal_anotherloss(model1, model2, node1, node2, num1, num2):
#     pos_MI = tf.constant([0], dtype=tf.float32)  # 正例
#     neg_MI = tf.constant([0], dtype=tf.float32)  # 负例
#     loss_list = []
#     for i in range(5):
#         graph_vec1 = tf.slice(model1, [i, 0], [1, P])  # 截取图嵌入
#         graph_vec2 = tf.slice(model2, [i, 0], [1, P])
#         MI_ori1 = tf.matmul(node1[i], tf.transpose(graph_vec2))  # 节点嵌入和图嵌入点积
#         MI_ori2 = tf.matmul(node2[i], tf.transpose(graph_vec1))
#
#         '''
#         张天豪修改　　　　１１．４
#         '''
#         MI_ori1 = get_positive_expectation(MI_ori1)
#         MI_ori2 = get_positive_expectation(MI_ori2)
#
#         MI_sum1 = tf.reduce_sum(MI_ori1, axis=0)  # 结果相加
#         MI_sum2 = tf.reduce_sum(MI_ori2, axis=0)
#
#         condition = tf.less(num1[i], num2[i])
#
#         node_num = tf.where(condition, num2[i], num1[i])  # 选出最大的节点数
#
#         MI1 = tf.divide(MI_sum1, node_num)  # 除以最大的节点数
#         MI2 = tf.divide(MI_sum2, node_num)
#
#         MI_fin = tf.add(MI1, MI2)
#         pos_MI = tf.add(MI_fin, pos_MI)  # 5个正例子加一起
#
#     for i in range(5, 10):
#         graph_vec1 = tf.slice(model1, [i, 0], [1, P])
#         graph_vec2 = tf.slice(model2, [i, 0], [1, P])
#         MI_ori1 = tf.matmul(node1[i], tf.transpose(graph_vec2))
#         MI_ori2 = tf.matmul(node2[i], tf.transpose(graph_vec1))
#
#         MI_ori1 = get_negative_expectation(MI_ori1)
#         MI_ori2 = get_negative_expectation(MI_ori2)
#
#         MI_sum1 = tf.reduce_sum(MI_ori1, axis=0)
#         MI_sum2 = tf.reduce_sum(MI_ori2, axis=0)
#
#         condition = tf.less(num1[i], num2[i])
#
#         node_num = tf.where(condition, num2[i], num1[i])
#
#         MI1 = tf.divide(MI_sum1, node_num)
#         MI2 = tf.divide(MI_sum2, node_num)
#
#         MI_fin = tf.add(MI1, MI2)
#         neg_MI = tf.add(MI_fin, neg_MI)
#
#     pos_loss = tf.divide(pos_MI, 5.0)
#     neg_loss = tf.divide(neg_MI, 5.0)
#
#     loss_list.append(pos_loss)
#     loss_list.append(neg_loss)
#
#     return neg_loss - pos_loss
#
#
# def cal_finloss(loss, another_loss):
#     '''
#     张天豪修改　　　　　１１．４
#     '''
#
#     W_3 = tf.get_variable('W_3', [1], tf.float32, initializer=tf.constant_initializer(0.001))
#
#     finloss = tf.add(loss, (W_3 * another_loss))
#
#     return finloss
#
#
# def plot_acc_loss(name, loss, acc):
#     host = host_subplot(111)  # row=1 col=1 first pic
#     plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
#     par1 = host.twinx()  # 共享x轴
#
#     # set labels
#     host.set_xlabel("steps")
#     host.set_ylabel(name + "-loss")
#     par1.set_ylabel(name + "-accuracy")
#
#     # plot curves
#     p1, = host.plot(range(len(loss)), loss, label="loss")
#     p2, = par1.plot(range(len(acc)), acc, label="accuracy")
#
#     # set location of the legend,
#     # 1->rightup corner, 2->leftup corner, 3->leftdown corner
#     # 4->rightdown corner, 5->rightmid ...
#     host.legend(loc=5)
#
#     # set label color
#     host.axis["left"].label.set_color(p1.get_color())
#     par1.axis["right"].label.set_color(p2.get_color())
#
#     # set the range of x axis of host and y axis of par1
#     # host.set_xlim([-200, 5200])
#     # par1.set_ylim([-0.1, 1.1])
#
#     plt.draw()
#     plt.show()
#
#
# def compute_accuracy(prediction, labels):
#     accu = 0.0
#     threshold = 0.5
#     for i in range(len(prediction)):
#         if labels[i][0] == 1:
#             if prediction[i][0] > threshold:
#                 accu += 1.0
#         else:
#             if prediction[i][0] < threshold:
#                 accu += 1.0
#     acc = accu / len(prediction)
#     return acc
#
#
# def cal_distance(model1, model2):
#     a_b = tf.reduce_sum(tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(model1, (1, -1)),
#                                                              tf.reshape(model2, (1, -1))], 0), 0), (B, P)), 1,
#                         keep_dims=True)
#     a_norm = tf.sqrt(tf.reduce_sum(tf.square(model1), 1, keep_dims=True))
#     b_norm = tf.sqrt(tf.reduce_sum(tf.square(model2), 1, keep_dims=True))
#     distance = a_b / tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(a_norm, (1, -1)),
#                                                           tf.reshape(b_norm, (1, -1))], 0), 0), (B, 1))
#     return distance
#
#
# def read_and_decode(filename):
#     # 根据文件名生成一个队列
#     filename_queue = tf.train.string_input_producer([filename])
#     # create a reader from file queue
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     # get feature from serialized example
#
#     features = tf.parse_single_example(serialized_example, features={
#         'label': tf.FixedLenFeature([], tf.int64),
#         'cfg_1': tf.FixedLenFeature([], tf.string),
#         'cfg_2': tf.FixedLenFeature([], tf.string),
#         'func_fea_1': tf.FixedLenFeature([], tf.string),
#         'func_fea_2': tf.FixedLenFeature([], tf.string),
#         'node_fea_1': tf.FixedLenFeature([], tf.string),
#         'node_fea_2': tf.FixedLenFeature([], tf.string),
#         'num1': tf.FixedLenFeature([], tf.int64),
#         'num2': tf.FixedLenFeature([], tf.int64),
#         'max': tf.FixedLenFeature([], tf.int64),
#         'inst_embedding1': tf.FixedLenFeature([], tf.string),
#         'inst_embedding2': tf.FixedLenFeature([], tf.string),
#         'inst_num1': tf.FixedLenFeature([], tf.string),
#         'inst_num2': tf.FixedLenFeature([], tf.string),
#         'w2v_1': tf.FixedLenFeature([], tf.string),
#         'w2v_2': tf.FixedLenFeature([], tf.string),
#         'w2v_num_1': tf.FixedLenFeature([], tf.string),
#         'w2v_num_2': tf.FixedLenFeature([], tf.string)
#     })
#
#
#     label = tf.cast(features['label'], tf.int32)
#
#     cfg_1 = features['cfg_1']
#     cfg_2 = features['cfg_2']
#     # result_fp.write(str(cfg_1))
#     # result_fp.write(str(cfg_2))
#
#
#     func_fea_1 = features['func_fea_1']
#     func_fea_2 = features['func_fea_2']
#
#     num1 = tf.cast(features['num1'], tf.int32)
#     node_fea_1 = features['node_fea_1']
#
#     num2 = tf.cast(features['num2'], tf.int32)
#     node_fea_2 = features['node_fea_2']
#     max_num = tf.cast(features['max'], tf.int32)
#
#     block_embedding1 = features['inst_embedding1']
#     block_embedding2 = features['inst_embedding2']
#     block_num1 = features['inst_num1']
#     block_num2 = features['inst_num2']
#     inst_embedding1 = features['w2v_1']
#     inst_embedding2 = features['w2v_2']
#     #
#     inst_num1 = features['w2v_num_1']
#     inst_num2 = features['w2v_num_2']
#     return label, cfg_1, cfg_2, func_fea_1, func_fea_2, node_fea_1, node_fea_2, num1, num2, max_num, \
#     block_embedding1, block_embedding2, inst_embedding1, inst_embedding2, inst_num1, inst_num2
#
# def cut_w2v(w2v,maxb,maxi):
#
#     w2v_ori = w2v
#     if w2v_ori.shape[1]>maxi:
#         w2v_ori = w2v_ori[:,:maxi,]
#
#     if w2v_ori.shape[1]<maxi:
#         zero_i = np.zeros([w2v_ori.shape[0],maxi - w2v_ori.shape[1],IE])
#         w2v_ori = np.concatenate([w2v_ori, zero_i], axis=1)
#
#     if w2v_ori.shape[0]>maxb:
#         w2v_ori = w2v_ori[:maxb]
#
#     if w2v_ori.shape[0]<maxb:
#         zero_b = np.zeros([maxb-w2v_ori.shape[0],MAX_I,IE])
#         w2v_ori = np.concatenate([w2v_ori, zero_b], axis=0)
#
#     return w2v_ori
#
# def get_batch(label, cfg_str1, cfg_str2, dfg_str1, dfg_str2, fea_str1, fea_str2, num1, num2, max_num, blockembed1,
#               blockembed2,instembed1, instembed2, instnum1, instnum2):
#     y = np.reshape(label, [B, 1])
#
#     v_num_1 = []
#     v_num_2 = []
#     for i in range(B):
#         v_num_1.append([int(num1[i])])
#         v_num_2.append([int(num2[i])])
#
#     # 补齐 martix 矩阵的长度
#     # cdfg_1 = []
#     # cdfg_2 = []
#     cfg_1 = []
#     cfg_2 = []
#     for i in range(B):
#         cfg_arr = np.array(cfg_str1[i].decode().split(','))
#         cfg_adj = np.reshape(cfg_arr, (int(num1[i]), int(num1[i])))  # reshape成邻接矩阵
#         cfg_ori1 = cfg_adj.astype(np.float32)
#         for node in range(int(num1[i])):
#             cfg_ori1[node][node] = 1.
#         # cfg only
#         cfg_zero1 = np.zeros([int(num1[i]), (int(max_num[i]) - int(num1[i]))])
#         cfg_zero2 = np.zeros([(int(max_num[i]) - int(num1[i])), int(max_num[i])])
#         cfg_vec1 = np.concatenate([cfg_ori1, cfg_zero1], axis=1)
#         cfg_vec2 = np.concatenate([cfg_vec1, cfg_zero2], axis=0)
#         cfg_1.append(cfg_vec2.tolist())
#
#         cfg_arr = np.array(cfg_str2[i].decode().split(','))
#         cfg_adj = np.reshape(cfg_arr, (int(num2[i]), int(num2[i])))
#         cfg_ori2 = cfg_adj.astype(np.float32)
#         for node in range(int(num2[i])):
#             cfg_ori2[node][node] = 1.
#
#         cfg_zero1 = np.zeros([int(num2[i]), (int(max_num[i]) - int(num2[i]))])
#         cfg_zero2 = np.zeros([(int(max_num[i]) - int(num2[i])), int(max_num[i])])
#         cfg_vec1 = np.concatenate([cfg_ori2, cfg_zero1], axis=1)
#         cfg_vec2 = np.concatenate([cfg_vec1, cfg_zero2], axis=0)
#         cfg_2.append(cfg_vec2.tolist())
#     # 补齐 feature 列表的长度
#     bert_1 = []
#     bert_2 = []
#     w2v_1 = []
#     w2v_2 = []
#     max_instnum1 = 0
#     for blk in instnum1:
#         for inst in np.array(blk.decode().split(',')).astype(np.int32):
#             if inst > max_instnum1:
#                 max_instnum1 = inst
#
#     max_instnum2 = 0
#     for blk in instnum2:
#         for inst in np.array(blk.decode().split(',')).astype(np.int32):
#             if inst > max_instnum2:
#                 max_instnum2 = inst
#
#     for i in range(B):
#         # fea_arr = np.array(fea_str1[i].split(','))
#         # fea_ori = fea_arr.astype(np.float32)
#         # fea_ori1 = np.resize(fea_ori, (int(num1[i]), D))
#
#         inst_embed_arr1 = np.array(instembed1[i].decode().split(','))
#         inst_embed_ori1 = inst_embed_arr1.astype(np.float32)
#         inst_embed_vec1 = np.reshape(inst_embed_ori1, [-1, IE])
#         inst_num_arr1 = np.array(instnum1[i].decode().split(','))
#         inst_num_ori1 = inst_num_arr1.astype(np.int64)
#         inst_func1 = []
#
#         flag1 = 0
#         for inst_n in inst_num_ori1:
#             # inst_func1.append(inst_embed_vec1[flag1:inst_n + flag1])
#             # flag1 = flag1 + inst_n
#             w2v_block_ori1 = inst_embed_vec1[flag1:inst_n + flag1]
#             inst_zero1 = np.zeros([int(max_instnum1 - inst_n), IE])
#             w2v_block_vec1 = np.concatenate([w2v_block_ori1, inst_zero1], axis=0)
#             inst_func1.append(w2v_block_vec1)
#             flag1 = flag1 + inst_n
#
#         inst_func1 = np.array(inst_func1)
#         block_zero1 = np.zeros([int(max_num[i] - num1[i]), int(max_instnum1), IE])
#         w2v_func_vec1 = np.concatenate([inst_func1,block_zero1])
#
#
#         w2v_1.append(cut_w2v(w2v_func_vec1,max_num[i],MAX_I).astype(np.float32))
#
#         block_embed_arr1 = np.array(blockembed1[i].decode().split(','))
#         block_embed_ori1 = block_embed_arr1.astype(np.float32)
#         block_embed_vec1 = np.reshape(block_embed_ori1, [-1, BE])
#
#         # fea_block_1 = np.concatenate([fea_ori1, block_embed_vec1], axis=1)
#         # fea_zero1 = np.zeros([int(max_num[i] - num1[i]), D + BE])
#         bert_zero1 = np.zeros([int(max_num[i] - block_embed_vec1.shape[0]),BE])
#         # fea_vec1 = np.concatenate([fea_block_1, fea_zero1], axis=0)
#         bert_vec1 = np.concatenate([block_embed_vec1, bert_zero1], axis=0)
#         bert_1.append(bert_vec1)
#
#         # fea_arr = np.array(fea_str2[i].split(','))
#         # fea_ori = fea_arr.astype(np.float32)
#         # fea_ori2 = np.resize(fea_ori, (int(num2[i]), D))
#
#         inst_embed_arr2 = np.array(instembed2[i].decode().split(','))
#         inst_embed_ori2 = inst_embed_arr2.astype(np.float32)
#         inst_embed_vec2 = np.reshape(inst_embed_ori2, [-1, IE])
#         inst_num_arr2 = np.array(instnum2[i].decode().split(','))
#         inst_num_ori2 = inst_num_arr2.astype(np.int64)
#         inst_func2 = []
#
#         flag2 = 0
#         for inst_n in inst_num_ori2:
#             # inst_func1.append(inst_embed_vec1[flag1:inst_n + flag1])
#             # flag1 = flag1 + inst_n
#             w2v_block_ori2 = inst_embed_vec2[flag2:inst_n + flag2]
#             inst_zero2 = np.zeros([int(max_instnum2 - inst_n), IE])
#             w2v_block_vec2 = np.concatenate([w2v_block_ori2, inst_zero2], axis=0)
#             inst_func2.append(w2v_block_vec2)
#             flag2 = flag2 + inst_n
#         inst_func2 = np.array(inst_func2)
#         block_zero2 = np.zeros([int(max_num[i] - num2[i]), int(max_instnum2), IE])
#         w2v_func_vec2 = np.concatenate([inst_func2,block_zero2])
#         w2v_2.append(cut_w2v(w2v_func_vec2,max_num[i],MAX_I).astype(np.float32))
#         # a = block_embed(w2v_2[0], 586, 100, 100)
#
#         block_embed_arr2 = np.array(blockembed2[i].decode().split(','))
#         block_embed_ori2 = block_embed_arr2.astype(np.float32)
#         block_embed_vec2 = np.reshape(block_embed_ori2, [-1, BE])
#         bert_zero2 = np.zeros([int(max_num[i] - block_embed_vec2.shape[0]), BE])
#         bert_vec2 = np.concatenate([block_embed_vec2, bert_zero2], axis=0)
#         bert_2.append(bert_vec2)
#
#     # a = block_embed(w2v_1[0][0], len(w2v_1[0][0]), 100, 100)
#
#     return y, cfg_1, cfg_2, bert_1, bert_2, v_num_1, v_num_2, w2v_1, w2v_2,max_instnum1,max_instnum2,max_num[0]#max_num is the same,which is 586
#
# def com_label_predict(labels,predicts,lables_predicts_file):
#     labels_r = np.reshape(labels, (-1))
#     predicts_r = np.reshape(predicts, (-1))
#
#     with open(lables_predicts_file, "w") as lpf:
#
#         for i in range(len(predicts_r)):
#             label = labels_r[i]
#
#             predict = predicts_r[i]
#
#             res_str = str(predict) +','+ str(label) + '\n'
#
#             lpf.write(res_str)
#
#     lpf.close()
#
#
# def write_results(csv_file, result_list):
#     with open(csv_file, "w") as fp:
#         for e in result_list:
#             res_str = str(e) + '\n'
#             fp.write(res_str)
#
#     fp.close()
#
# def get_datasetnum(csv_file, program_info):
#
#     test_dataset_num = 0
#     train_dataset_num = 0
#     valid_dataset_num = 0
#
#     for csv_name in os.listdir(csv_file):
#         csv_name_list = csv_name.split('_')
#         csv_info = (csv_name_list[1], csv_name_list[2], csv_name_list[3], csv_name_list[4])
#         if csv_name_list[0] == "test" and csv_info == program_info:
#             with open(csv_file + os.sep + csv_name, 'r') as f:
#                 test_dataset_num = len(f.readlines())
#
#         if csv_name_list[0] == "train" and csv_info == program_info:
#             with open(csv_file + os.sep + csv_name, 'r') as f:
#                 train_dataset_num = len(f.readlines())
#
#         if csv_name_list[0] == "valid" and csv_info == program_info:
#             with open(csv_file + os.sep + csv_name, 'r') as f:
#                 valid_dataset_num = len(f.readlines())
#     # print 'asdfg'
#     return train_dataset_num, valid_dataset_num, test_dataset_num
# # 4.construct the network
# # Initializing the variables
# # Siamese network major part
#
# # Initializing the variables
#
# init = tf.global_variables_initializer()
# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase=True)
#
# v_num_left = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_left')
# cdfg_left = tf.placeholder(tf.float32, shape=([B, None, None]), name='cdfg_left')
# # fea_left = tf.placeholder(tf.float32, shape=([B, None, D + BE + IE]), name='fea_left')
# fea_left = tf.placeholder(tf.float32, shape=([B, None, BE ]), name='fea_left')
# w2v_left = tf.placeholder(tf.float32, shape=([B, None, None, IE ]), name='w2v_left')
# inst_max_left = tf.placeholder('int32')
#
# v_num_right = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_right')
# cdfg_right = tf.placeholder(tf.float32, shape=([B, None, None]), name='cdfg_right')
# # fea_right = tf.placeholder(tf.float32, shape=([B, None, D + BE + IE]), name='fea_right')
# fea_right = tf.placeholder(tf.float32, shape=([B, None, BE ]), name='fea_right')
# w2v_right = tf.placeholder(tf.float32, shape=([B, None, None,IE ]), name='w2v_right')
# inst_max_right = tf.placeholder('int32')
#
# max_num = tf.placeholder('int32')
# labels = tf.placeholder(tf.float32, shape=([B, 1]), name='gt')
#
# dropout_f = tf.placeholder("float")
#
#
# with tf.variable_scope("siamese") as siamese:
#
#
#     model1 = structure2vec_net(cdfg_left, fea_left, w2v_left)#(cdfgs, block_eb, inst_eb,max_num,max_inst_num)
#     siamese.reuse_variables()
#     model2 = structure2vec_net(cdfg_right, fea_right ,w2v_right)
#     # model1 = safe_net(w2v_left)
#     # siamese.reuse_variables()
#     # model2 = safe_net(w2v_right)
#
# dis = cal_distance(model1, model2)
#
# loss = contrastive_loss(labels, dis)
#
# # another_loss = cal_anotherloss(model1,model2,node1,node2,num1,num2)
#
# # finloss = cal_finloss(loss,another_loss)
#
#
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
# # 三大数据：基本块特征，控制流图，数据流图
# # list_train_label：
# # cfg :控制流图 dfg:数据流图  fea:基本块特征
# ''
#
# list_train_label, list_train_cfg_1, list_train_cfg_2, list_train_func_fea_1, list_train_func_fea_2, list_train_node_fea_1, \
# list_train_node_fea_2, list_train_num1, list_train_num2, list_train_max, \
# list_train_block_embedding1, list_train_block_embedding2, \
# list_train_inst_embedding1, list_train_inst_embedding2, list_train_inst_num1, list_train_inst_num2 \
#     = read_and_decode(TRAIN_TFRECORD)
#
# batch_train_label, batch_train_cfg_1, batch_train_cfg_2, batch_train_func_fea_1, batch_train_func_fea_2, batch_train_node_fea_1, \
# batch_train_node_fea_2, batch_train_num1, batch_train_num2, batch_train_max, \
# batch_train_block_embedding1, batch_train_block_embedding2, \
# batch_train_inst_embedding1, batch_train_inst_embedding2, \
# batch_train_inst_num1, batch_train_inst_num2 \
#     = tf.train.batch(
#     [list_train_label, list_train_cfg_1, list_train_cfg_2, list_train_func_fea_1, list_train_func_fea_2,
#      list_train_node_fea_1, list_train_node_fea_2, list_train_num1, list_train_num2, list_train_max,
#      list_train_block_embedding1, list_train_block_embedding2,
#      list_train_inst_embedding1, list_train_inst_embedding2, list_train_inst_num1, list_train_inst_num2],
#     batch_size=B, capacity=10)
#
# list_valid_label, list_valid_cfg_1, list_valid_cfg_2, list_valid_dfg_1, list_valid_dfg_2, list_valid_fea_1, \
# list_valid_fea_2, list_valid_num1, list_valid_num2, list_valid_max, \
# list_valid_block_embedding1, list_valid_block_embedding2, \
# list_valid_inst_embedding1, list_valid_inst_embedding2, list_valid_inst_num1, list_valid_inst_num2 \
#     = read_and_decode(VALID_TFRECORD)
#
# batch_valid_label, batch_valid_cfg_1, batch_valid_cfg_2, batch_valid_dfg_1, batch_valid_dfg_2, batch_valid_fea_1, \
# batch_valid_fea_2, batch_valid_num1, batch_valid_num2, batch_valid_max, \
# batch_valid_block_embedding1, batch_valid_block_embedding2, \
# batch_valid_inst_embedding1, batch_valid_inst_embedding2, batch_valid_inst_num1, batch_valid_inst_num2 \
#     = tf.train.batch([list_valid_label, list_valid_cfg_1, list_valid_cfg_2, list_valid_dfg_1, list_valid_dfg_2,
#                       list_valid_fea_1, list_valid_fea_2, list_valid_num1, list_valid_num2, list_valid_max,
#                       list_valid_block_embedding1, list_valid_block_embedding2,
#                       list_valid_inst_embedding1, list_valid_inst_embedding2, list_valid_inst_num1,
#                       list_valid_inst_num2],
#                      batch_size=B, capacity=10)
#
# list_test_label, list_test_cfg_1, list_test_cfg_2, list_test_dfg_1, list_test_dfg_2, list_test_fea_1, \
# list_test_fea_2, list_test_num1, list_test_num2, list_test_max, \
# list_test_block_embedding1, list_test_block_embedding2, \
# list_test_inst_embedding1, list_test_inst_embedding2, list_test_test_num1, list_test_test_num2 \
#     = read_and_decode(TEST_TFRECORD)
#
# batch_test_label, batch_test_cfg_1, batch_test_cfg_2, batch_test_dfg_1, batch_test_dfg_2, batch_test_fea_1, \
# batch_test_fea_2, batch_test_num1, batch_test_num2, batch_test_max, \
# batch_test_block_embedding1, batch_test_block_embedding2, \
# batch_test_inst_embedding1, batch_test_inst_embedding2, batch_test_inst_num1, batch_test_inst_num2 \
#     = tf.train.batch([list_test_label, list_test_cfg_1, list_test_cfg_2, list_test_dfg_1, list_test_dfg_2,
#                       list_test_fea_1, list_test_fea_2, list_test_num1, list_test_num2, list_test_max,
#                       list_test_block_embedding1, list_test_block_embedding2,
#                       list_test_inst_embedding1, list_test_inst_embedding2, list_test_test_num1, list_test_test_num2],
#                      batch_size=B, capacity=10)
#
# ''''''
# init_opt = tf.global_variables_initializer()
# saver = tf.train.Saver()
# #
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# # config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
# tfconfig = tf.ConfigProto()
# tfconfig.gpu_options.allow_growth = True
# # with tf.Session(config=config) as sess:
#
# train_acc_list = []
# train_loss_list = []
# test_acc_list = []
# test_loss_list = []
#
# with tf.Session(config=tfconfig) as sess:
#     # with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:
#     # writer = tf.summary.FileWriter('logs/', sess.graph)
#     sess.run(init_opt)
#     # flag = 1
#     # for example in tf.python_io.tf_record_iterator(TRAIN_TFRECORD):
#     #     if flag == 895:
#     #         print flag
#     #         x = str(tf.train.Example.FromString(example))
#     #         result_fp.write("zheshiyigehanshu" + str(flag) + x)
#     #         result_fp.write("\n")
#     #         break
#     #     else:
#     #         flag = flag + 1
#     train_num, valid_num, test_num = get_datasetnum(csv_file, program_info)
#     print(train_num, valid_num, test_num)
#     precision_list = []
#     AUC_list = []
#     recall_list = []
#     accuracy_list = []
#     f1score_list = []
#
#     train_label_list = []
#     train_predict_list = []
#     test_label_list = []
#     test_predict_list = []
#     #
#     #     flag = flag + 1
#     # if config.SETP5_IF_RESTORE_VULSEEKER_MODEL:
#     #     saver.restore(sess, config.MODEL_VULSEEKER_DIR + os.sep + config.STEP5_VULSEEKER_MODEL_TO_RESTORE)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     # Training cycle
#     iter = 0
#     while iter < max_iter:
#         iter += 1
#         avg_loss = 0.
#         avg_acc = 0.
#         total_batch = int(train_num / B)
#         start_time = time.time()
#         # Loop over all batches
#         # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_nu
#         print (total_batch)
#         for i in range(total_batch):  # 原数据：total_batch
#
#             train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1, train_node_fea_2, \
#             train_num1, train_num2, train_max, \
#             train_block_embedding1, train_block_embedding2, \
#             train_inst_embedding1, train_inst_embedding2, train_inst_num1, train_inst_num2 \
#                 = sess.run([batch_train_label, batch_train_cfg_1, batch_train_cfg_2, batch_train_func_fea_1,
#                             batch_train_func_fea_2, batch_train_node_fea_1, batch_train_node_fea_2, batch_train_num1,
#                             batch_train_num2, batch_train_max,
#                             batch_train_block_embedding1, batch_train_block_embedding2,
#                             batch_train_inst_embedding1, batch_train_inst_embedding2,
#                             batch_train_inst_num1, batch_train_inst_num2])
#
#             y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2, w2v_1, w2v_2,inst_max1,inst_max2,max_block \
#                 = get_batch(train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1,
#                             train_node_fea_2,
#                             train_num1, train_num2, train_max,
#                             train_block_embedding1, train_block_embedding2,
#                             train_inst_embedding1, train_inst_embedding2, train_inst_num1, train_inst_num2)
#             # MAX_B = max_block
#             # MAX_I1 = inst_max1
#             # MAX_I2 = inst_max2
#             _, loss_value, predict = sess.run([optimizer, loss, dis], feed_dict={
#                 w2v_left: w2v_1, cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1,
#                 w2v_right: w2v_2, cdfg_right: cdfg_2, fea_right: fea_2,v_num_right: v_num_2,
#                 max_num:max_block,
#                 labels: y, dropout_f: 0.9})
#             tr_acc = compute_accuracy(predict, y)
#             # train_acc_list.append(tr_acc)
#             # train_loss_list.append(loss_value)
#             if is_debug:
#                 print ('     %d    tr_acc %0.2f    tr_loss  %0.2f' % (i, tr_acc, loss_value))
#                 # sys.stdout.flush()
#             avg_loss += loss_value
#             avg_acc += tr_acc * 100
#         duration = time.time() - start_time
#
#         if iter % snapshot == 0:
#             # validing model
#             avg_loss = 0.
#             avg_acc = 0.
#             valid_start_time = time.time()
#             print (int(valid_num / B))
#             for m in range(int(valid_num / B)):  # 原数据：int(valid_num / B)
#                 valid_label, valid_cfg_1, valid_cfg_2, valid_func_fea_1, valid_func_fea_2, valid_node_fea_1, valid_node_fea_2, \
#                 valid_num1, valid_num2, valid_max, \
#                 valid_block_embedding1, valid_block_embedding2, \
#                 valid_inst_embedding1, valid_inst_embedding2, valid_inst_num1, valid_inst_num2 \
#                     = sess.run([batch_valid_label, batch_valid_cfg_1, batch_valid_cfg_2, batch_valid_dfg_1,
#                                 batch_valid_dfg_2, batch_valid_fea_1, batch_valid_fea_2, batch_valid_num1,
#                                 batch_valid_num2, batch_valid_max,
#                                 batch_valid_block_embedding1, batch_valid_block_embedding2,
#                                 batch_valid_inst_embedding1, batch_valid_inst_embedding2, batch_valid_inst_num1,
#                                 batch_valid_inst_num2])
#
#                 y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 , w2v_1, w2v_2,inst_max1,inst_max2,max_block\
#                     = get_batch(valid_label, valid_cfg_1, valid_cfg_2, valid_func_fea_1, valid_func_fea_2,
#                                 valid_node_fea_1, valid_node_fea_2,
#                                 valid_num1, valid_num2, valid_max,
#                                 valid_block_embedding1, valid_block_embedding2,
#                                 valid_inst_embedding1, valid_inst_embedding2, valid_inst_num1, valid_inst_num2)
#                 predict = dis.eval(feed_dict={
#                 w2v_left: w2v_1, cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1,
#                 w2v_right: w2v_2, cdfg_right: cdfg_2, fea_right: fea_2,v_num_right: v_num_2,
#                 max_num:max_block,
#                 labels: y, dropout_f: 0.9})
#                 tr_acc = compute_accuracy(predict, y)
#                 valid_loss = loss.eval(feed_dict={labels: y, dis: predict})
#                 avg_loss += valid_loss
#                 avg_acc += tr_acc * 100
#                 if is_debug:
#                     print ('valid     tr_acc %0.2f    tr_loss %0.2f  '%(tr_acc,valid_loss))
#             duration = time.time() - valid_start_time
#             # print 'valid set, %d,  time, %f, loss, %0.5f, acc, %0.2f' % (
#             #    iter, duration, avg_loss / (int(valid_num / B)), avg_acc / (int(valid_num / B)))
#             # saver.save(sess,
#             #            config.MODEL_VULSEEKER_DIR + os.sep + "vulseeker-model" + PREFIX + "_" + str(iter) + "bert_only.ckpt")
#             if iter == max_iter:
#                 fpr_file = config.BERT_RESULT_DIR + os.sep + PREFIX + '_' + str(iter) + '_fpr.csv'
#                 tpr_file = config.BERT_RESULT_DIR + os.sep + PREFIX + '_' + str(iter) + '_tpr.csv'
#             total_labels = []
#             total_predicts = []
#             avg_loss = 0.
#             avg_acc = 0.
#             test_total_batch = int(test_num / B)
#             start_time = time.time()
#             # Loop over all batches
#             # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_num
#             print (test_total_batch)
#             for m in range(test_total_batch):  # 原数据：test_total_batch
#                 test_label, test_cfg_1, test_cfg_2, test_func_fea_1, test_func_fea_2, \
#                 test_node_fea_1, test_node_fea_2, test_num1, test_num2, test_max, \
#                 test_block_embedding1, test_block_embedding2, \
#                 test_inst_embedding1, test_inst_embedding2, test_inst_num1, test_inst_num2 \
#                     = sess.run(
#                     [batch_test_label, batch_test_cfg_1, batch_test_cfg_2, batch_test_dfg_1, batch_test_dfg_2,
#                      batch_test_fea_1, batch_test_fea_2, batch_test_num1, batch_test_num2, batch_test_max,
#                      batch_test_block_embedding1, batch_test_block_embedding2,
#                      batch_test_inst_embedding1, batch_test_inst_embedding2, batch_test_inst_num1,
#                      batch_test_inst_num2])
#
#                 y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 ,w2v_1, w2v_2,inst_max1,inst_max2,max_block \
#                     = get_batch(test_label, test_cfg_1, test_cfg_2, test_func_fea_1, test_func_fea_2,
#                                 test_node_fea_1, test_node_fea_2, test_num1, test_num2, test_max,
#                                 test_block_embedding1, test_block_embedding2,
#                                 test_inst_embedding1, test_inst_embedding2, test_inst_num1, test_inst_num2)
#
#                 predict = dis.eval(feed_dict={
#                 w2v_left: w2v_1, cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1,
#                 w2v_right: w2v_2, cdfg_right: cdfg_2, fea_right: fea_2,v_num_right: v_num_2,
#                 max_num:max_block,
#                 labels: y, dropout_f: 0.9})
#                 tr_acc = compute_accuracy(predict, y)
#                 test_loss = loss.eval(feed_dict={labels: y, dis: predict})
#                 avg_loss += test_loss
#                 avg_acc += tr_acc * 100
#                 total_labels.append(y)
#                 total_predicts.append(predict)
#                 if iter == max_iter:
#                     test_label_list.append(y)
#                     test_predict_list.append(predict)
#                 if is_debug:
#                     print ('test     %d    tr_acc %0.2f   tr_loss %0.2f' % (m, tr_acc,test_loss))
#             duration = time.time() - start_time
#             total_labels = np.reshape(total_labels, (-1))
#             total_predicts = np.reshape(total_predicts, (-1))
#             fpr, tpr, AUC, precision, recall, f1score = calculate_auc(total_labels, total_predicts)
#             print (AUC)
#             print ('test set, time, %f, loss, %0.5f, acc, %0.2f' % (
#             duration, avg_loss / test_total_batch, avg_acc / test_total_batch))
#             test_loss_list.append(avg_loss/ test_total_batch)
#             test_acc_list.append(avg_acc/ test_total_batch)
#
#             AUC_list.append(AUC)
#             precision_list.append(precision)
#             recall_list.append(recall)
#             accuracy_list.append((avg_acc / test_total_batch))
#             f1score_list.append(f1score)
#
#             if iter == max_iter:
#                 write_results(fpr_file, fpr)
#                 write_results(tpr_file, tpr)
#
#
#     print ("end")
#
#     write_results(accuracy_file, accuracy_list)
#     write_results(precision_file, precision_list)
#     write_results(AUC_file, AUC_list)
#     write_results(f1score_file, f1score_list)
#     write_results(recall_file, recall_list)
#
#     com_label_predict(train_label_list,train_predict_list,train_label_predict_file)
#     com_label_predict(test_label_list,test_predict_list,test_label_predict_file)
#     plot_acc_loss('test',test_loss_list,test_acc_list)
#
#     # 保存模型
#     # saver.save(sess, config.MODEL_VULSEEKER_DIR + os.sep + "vulseeker-model" + PREFIX + "_final_bert_only.ckpt")
#
#     # coord.request_stop()
#     # coord.join(threads)
