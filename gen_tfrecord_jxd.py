#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import json
import numpy as np
import networkx as nx
import itertools
import tensorflow as tf

import config_jxd as config
import os
import csv
import glob

# program_name = config.program_name
# program_version = config.program_version
# program_arch = config.program_arch
# program_opti = config.program_opti
# program_info = (program_name, program_version, program_arch, program_opti)


npy_file = './5_block_embedding'
inst_npyfile = './7_w2v_ori_embedding/clean_w2v_func_vec'
json_dir = './6_json_files'

# PREFIX = program_name + '_' + program_version + '_' + program_arch + '_' + program_opti

# train_file = config.DATASET_DIR + os.sep + "train_"+ PREFIX+"_t.csv"
# test_file = config.DATASET_DIR + os.sep + "test_"+ PREFIX+"_t.csv"
# valid_file = config.DATASET_DIR + os.sep + "vaild_"+ PREFIX+"_t.csv"
#
# TRAIN_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "train_" + PREFIX + ".tfrecord"
# TEST_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "test_" + PREFIX + ".tfrecord"
# VALID_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "valid_" + PREFIX + ".tfrecord"

is_debug = True


def load_dataset():
    train_pair,train_label = load_csv_as_pair(train_file)
    valid_pair,valid_label = load_csv_as_pair(valid_file)
    test_pair,test_label = load_csv_as_pair(test_file)

    return train_pair,train_label,valid_pair,valid_label,test_pair,test_label


def load_csv_as_pair(pair_label_file):

    pair_list = []
    label_list = []
    with open(pair_label_file,"r") as fp:
        pair_label = csv.reader(fp)
        for line in pair_label:
            pair_list.append([line[0],line[1],line[2],line[3]])
            label_list.append(int(line[4]))

    return pair_list,label_list



def construct_learning_dataset(uid_pair_list):


    print "     start generate features pairs..."
    ### !!! record the max number of a function's block
    func_feas_1, func_feas_2, node_feas_1, node_feas_2, max_size, num1, num2 = generate_features_pair(uid_pair_list)

    print "     start generate bertEmbedding pairs..."
    inst_embedding1, inst_embedding2, inst_num1, inst_num2 = generate_bert_Embedding_pair(uid_pair_list)

    print "     start generate w2vEmbedding pairs..."
    # inst_embedding1, inst_embedding2, inst_num1, inst_num2 = generate_instEmbedding_pair(uid_pair_list)
    w2v_embedding1, w2v_embedding2, w2v_num1, w2v_num2 = generate_instEmbedding_pair(uid_pair_list)


    print "     start generate adj matrix pairs..."

    cfgs_1, cfgs_2 = generate_graph_pair(uid_pair_list)


    return cfgs_1, cfgs_2, func_feas_1, func_feas_2, node_feas_1, node_feas_2, num1, num2, max_size, inst_embedding1, inst_embedding2, inst_num1, inst_num2, \
    w2v_embedding1, w2v_embedding2, w2v_num1, w2v_num2
    # return cfgs_1, cfgs_2, func_feas_1, func_feas_2, node_feas_1, node_feas_2, num1, num2, max_size


def generate_graph_pair(uid_pair_list):
    cfgs_1 = []
    cfgs_2 = []

    count = 0

    for uid_pair in uid_pair_list:
        if is_debug:
            count += 1
            print "         %04d cfg, [ %s %s  ,  %s %s]"%(count, uid_pair[0], uid_pair[1],uid_pair[2],uid_pair[3])

        json_path1 = json_dir + os.sep + uid_pair[0].split(os.sep)[-1]

        with open(json_path1,"r") as json1:
            cfg_ori_data1 = []

            for line1 in json1:
                data1 = json.loads(line1)
                if data1['fname'] == uid_pair[1]:
                    cfg_ori_data1.extend(data1['succs'])
                    break

        node1 = [x for x in range(len(cfg_ori_data1))]
        graph_cfg1 = nx.Graph()
        graph_cfg1.add_nodes_from(node1)

        for graph_node1 in graph_cfg1.nodes():
            edge_list1 = cfg_ori_data1[graph_node1]
            for edge_node1 in edge_list1:
                graph_cfg1.add_edge(graph_node1,edge_node1)
        adj_arr1 = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg1,dtype=float))
        adj_str1 = adj_arr1.astype(np.string_)
        cfgs_1.append(",".join(list(itertools.chain.from_iterable(adj_str1))))





        json_path2 =  json_dir + os.sep + uid_pair[2].split(os.sep)[-1]
        with open(json_path2, "r") as json2:
            cfg_ori_data2 = []

            for line2 in json2:
                data2 = json.loads(line2)
                if data2['fname'] == uid_pair[3]:
                    cfg_ori_data2.extend(data2['succs'])
                    break

        node2 = [x for x in range(len(cfg_ori_data2))]
        graph_cfg2 = nx.Graph()
        graph_cfg2.add_nodes_from(node2)

        for graph_node2 in graph_cfg2.nodes():
            edge_list2 = cfg_ori_data2[graph_node2]
            for edge_node2 in edge_list2:
                graph_cfg2.add_edge(graph_node2, edge_node2)
        adj_arr2 = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg2, dtype=float))
        adj_str2 = adj_arr2.astype(np.string_)
        cfgs_2.append(",".join(list(itertools.chain.from_iterable(adj_str2))))


    return cfgs_1,cfgs_2



def generate_features_pair(uid_pair_list):
    func_feas_1 = []
    func_feas_2 = []
    node_feas_1 = []
    node_feas_2 = []
    num1 = []
    num2 = []
    node_length = []

    count = 0
    for uid_pair in uid_pair_list:
        if is_debug:
            count += 1
            print "         %04d fea, [ %s %s  ,  %s %s]"%(count, uid_pair[0], uid_pair[1],uid_pair[2],uid_pair[3])

        json_path1 =  json_dir + os.sep + uid_pair[0].split(os.sep)[-1]

        with open(json_path1,'r') as json1:
            node_ori_1 = []
            func_ori_1 = []
            for line1 in json1.readlines():
                data1 = json.loads(line1)
                if data1['fname'] == uid_pair[1]:
                    node_ori_1.extend(data1['features'])
                    func_ori_1.extend(data1['func_feature'])
                    num1.append(data1['n_num'])
                    node_length.append(data1['n_num'])
                    break

            node_vector_1 = []
            for node_ori1 in node_ori_1:
                block_feature1 = [float(x) for x in (node_ori1[:11])]
                node_vector_1.append(block_feature1)

            func_vector_1 = []
            func_feature1 = [float(x) for x in (func_ori_1)]
            func_vector_1.append(func_feature1)




        node_arr1 = np.array(node_vector_1)
        node_str1 = node_arr1.astype(np.string_)
        node_feas_1.append(",".join(list(itertools.chain.from_iterable(node_str1))))


        func_arr1 = np.array(func_vector_1)
        func_str1 = func_arr1.astype(np.string_)
        func_feas_1.append(" ".join(list(itertools.chain.from_iterable(func_str1))))





        json_path2 =  json_dir + os.sep + uid_pair[2].split(os.sep)[-1]

        with open(json_path2, 'r') as json2:
            node_ori_2 = []
            func_ori_2 = []
            for line2 in json2.readlines():
                data2 = json.loads(line2)
                if data2['fname'] == uid_pair[3]:
                    node_ori_2.extend(data2['features'])
                    func_ori_2.extend(data2['func_feature'])
                    num2.append(data2['n_num'])
                    node_length.append(data2['n_num'])
                    break

            node_vector_2 = []
            for node_ori2 in node_ori_2:
                block_feature2 = [float(x) for x in (node_ori2[:11])]
                node_vector_2.append(block_feature2)

            func_vector_2 = []
            func_feature2 = [float(x) for x in (func_ori_2)]
            func_vector_2.append(func_feature2)

        node_arr2 = np.array(node_vector_2)
        node_str2 = node_arr2.astype(np.string_)
        node_feas_2.append(",".join(list(itertools.chain.from_iterable(node_str2))))

        func_arr2 = np.array(func_vector_2)
        func_str2 = func_arr2.astype(np.string_)
        func_feas_2.append(",".join(list(itertools.chain.from_iterable(func_str2))))

    num1_re = np.array(num1)
    num2_re = np.array(num2)


    return func_feas_1, func_feas_2, node_feas_1, node_feas_2, np.max(node_length), num1_re, num2_re



# def generate_w2v_Embedding_pair(uid_pair_list):
#     # instEmbedding_ori = np.load(npy_file)
#
#     inst_embedding1 = []
#     inst_embedding2 = []
#     inst_num_list1 = []
#     inst_num_list2 = []
#
#     count = 0
#
#     for uid_pair in uid_pair_list:
#         if is_debug:
#             count+= 1
#             print "         %04d w2v, [ %s %s  ,  %s %s]"%(count, uid_pair[0], uid_pair[1],uid_pair[2],uid_pair[3])
#
#         inst_vector = []
#         inst_num = []
#         uid_pair_sp1 = uid_pair[0].split('_')
#         npy_file1 =inst_npyfile + os.sep + uid_pair[0].split(os.sep)[-1][:-5] + os.sep + uid_pair[1]+".npy"
#         instEmbedding_ori1 = np.load(npy_file1,allow_pickle=True)
#
#         for instEmbedding in instEmbedding_ori1:
#             inst_vector.append(instEmbedding)
#
#
#         inst_arr = np.array(inst_vector)
#         inst_con = np.concatenate(inst_arr, axis=0)
#         inst_str1 = str(list(inst_con)).replace('[', '').replace(']', '')
#         inst_num_list1.append(str(len(inst_arr)))
#         # num_arr = np.array(inst_num)
#         # inst_str = inst_con.astype(np.string_)
#         # num_arr1 = num_arr.astype(np.string_)
#         inst_embedding1.append(inst_str1)
#         # inst_num_list1.append(num_arr1)
#
#         inst_vector = []
#         inst_num = []
#         uid_pair_sp2 = uid_pair[2].split('_')
#         # npy_file2 = npy_file + os.sep + uid_pair_sp2[1] + "_" + uid_pair_sp2[2] + "_" + uid_pair_sp2[3] + os.sep + uid_pair[3] + ".npy"
#         npy_file2 = inst_npyfile + os.sep + uid_pair[2].split(os.sep)[-1][:-5]  + os.sep + uid_pair[3]+".npy"
#         instEmbedding_ori2 = np.load(npy_file2,allow_pickle=True)
#         for instEmbedding in instEmbedding_ori2:
#             inst_vector.append(instEmbedding)
#
#
#         inst_arr = np.array(inst_vector)
#         inst_con = np.concatenate(inst_arr, axis=0)
#         # inst_list = []
#         inst_str2 = str(list(inst_con)).replace('[', '').replace(']', '')
#         inst_num_list2.append(str(len(inst_arr)))
#         # inst_num_list2.append(str(len(inst_arr)))
#         # num_arr = np.array(inst_num)
#         # inst_str = inst_con.astype(np.string_)
#         # num_arr2 = num_arr.astype(np.string_)
#         inst_embedding2.append(inst_str2)
#         # inst_num_list2.append(','.join(list(itertools.chain.from_iterable(num_arr))))
#         # inst_num_list2.append(','.join(list(itertools.chain.from_iterable(num_arr2))))
#
#     return inst_embedding1,inst_embedding2,inst_num_list1,inst_num_list2


def generate_instEmbedding_pair(uid_pair_list):
    # instEmbedding_ori = np.load(npy_file)

    inst_embedding1 = []
    inst_embedding2 = []
    inst_num_list1 = []
    inst_num_list2 = []

    count = 0

    for uid_pair in uid_pair_list:
        if is_debug:
            count+= 1
            print "         %04d inst, [ %s %s  ,  %s %s]"%(count, uid_pair[0], uid_pair[1],uid_pair[2],uid_pair[3])

        inst_vector = []
        inst_num = []
        uid_pair_sp1 = uid_pair[0].split('_')
        npy_file1 = inst_npyfile+ os.sep + uid_pair[0].split(os.sep)[-1][:-5] + os.sep + uid_pair[1]+".npy"
        instEmbedding_ori1 = np.load(npy_file1,allow_pickle=True)

        for instEmbedding in instEmbedding_ori1:
            inst_vector.append(instEmbedding)
            inst_num.append(len(instEmbedding))

        inst_arr = np.array(inst_vector)
        inst_con = np.concatenate(inst_arr, axis=0)
        num_arr = np.array(inst_num)
        inst_str = inst_con.astype(np.string_)
        num_arr = num_arr.astype(np.string_)
        inst_embedding1.append(','.join(list(itertools.chain.from_iterable(inst_str))))
        inst_num_list1.append(','.join(list(itertools.chain.from_iterable(num_arr))))


        inst_vector = []
        inst_num = []
        npy_file2 = inst_npyfile + os.sep + uid_pair[2].split(os.sep)[-1][:-5] + os.sep + uid_pair[3] + ".npy"
        instEmbedding_ori2 = np.load(npy_file2,allow_pickle=True)
        for instEmbedding in instEmbedding_ori2:
            inst_vector.append(instEmbedding)
            inst_num.append(len(instEmbedding))

        inst_arr = np.array(inst_vector)
        num_arr = np.array(inst_num)
        inst_con = np.concatenate(inst_arr, axis=0)
        inst_str = inst_con.astype(np.string_)
        num_arr = num_arr.astype(np.string_)
        inst_embedding2.append(','.join(list(itertools.chain.from_iterable(inst_str))))
        inst_num_list2.append(','.join(list(itertools.chain.from_iterable(num_arr))))


    return inst_embedding1,inst_embedding2,inst_num_list1,inst_num_list2



def generate_bert_Embedding_pair(uid_pair_list):
    # instEmbedding_ori = np.load(npy_file)

    inst_embedding1 = []
    inst_embedding2 = []
    inst_num_list1 = []
    inst_num_list2 = []

    count = 0

    for uid_pair in uid_pair_list:
        if is_debug:
            count+= 1
            print "         %04d bert, [ %s %s  ,  %s %s]"%(count, uid_pair[0], uid_pair[1],uid_pair[2],uid_pair[3])

        inst_vector = []
        inst_num = []
        # uid_pair_sp1 = uid_pair[0].split('_')
        # npy_file1 =npy_file+os.sep+uid_pair_sp1[1]+"_"+uid_pair_sp1[2]+"_"+uid_pair_sp1[3]+os.sep+uid_pair[1]+".npy"
        npy_file1 = npy_file+ os.sep + uid_pair[0].split(os.sep)[-1][:-5] + os.sep + uid_pair[1]+".npy"

        instEmbedding_ori1 = np.load(npy_file1,allow_pickle=True)

        for instEmbedding in instEmbedding_ori1:
            inst_vector.append(instEmbedding)

        inst_arr = np.array(inst_vector)
        inst_con = np.concatenate(inst_arr, axis=0)
        inst_str1 = str(list(inst_con)).replace('[', '').replace(']', '')
        inst_num_list1.append(str(len(inst_arr)))
        inst_embedding1.append(inst_str1)

        # inst_arr = np.array(inst_vector)
        # inst_con = np.concatenate(inst_arr, axis=0)
        # inst_num.append(inst_con.shape[0])
        # num_arr = np.array(inst_num)
        # inst_str = inst_con.astype(np.string_)
        # num_arr = num_arr.astype(np.string_)
        # inst_embedding1.append(','.join(list(itertools.chain.from_iterable(inst_str))))
        #
        # inst_num_list1.append(','.join(list(itertools.chain.from_iterable(num_arr))))


        inst_vector = []
        inst_num = []
        # uid_pair_sp2 = uid_pair[2].split('_')
        # npy_file2 = npy_file + os.sep + uid_pair_sp2[1] + "_" + uid_pair_sp2[2] + "_" + uid_pair_sp2[3] + os.sep + uid_pair[3] + ".npy"
        npy_file2 = npy_file + os.sep + uid_pair[2].split(os.sep)[-1][:-5]  + os.sep + uid_pair[3]+".npy"
        instEmbedding_ori2 = np.load(npy_file2,allow_pickle=True)
        for instEmbedding in instEmbedding_ori2:
            inst_vector.append(instEmbedding)

        inst_arr = np.array(inst_vector)
        inst_con = np.concatenate(inst_arr, axis=0)
        inst_str2 = str(list(inst_con)).replace('[', '').replace(']', '')
        inst_num_list2.append(str(len(inst_arr)))
        inst_embedding2.append(inst_str2)
        # inst_arr = np.array(inst_vector)
        # inst_con = np.concatenate(inst_arr, axis=0)
        # inst_num.append(inst_con.shape[0])
        # num_arr = np.array(inst_num)
        # inst_str = inst_con.astype(np.string_)
        # num_arr = num_arr.astype(np.string_)
        # inst_embedding2.append(','.join(list(itertools.chain.from_iterable(inst_str))))
        # inst_num_list2.append(','.join(list(itertools.chain.from_iterable(num_arr))))

    return inst_embedding1,inst_embedding2,inst_num_list1,inst_num_list2




if __name__ == '__main__':

    files = glob.glob('./2_NewDataset/func_list' + os.sep + 'test' + '*')
    for file in files:

        detail = file.split(os.sep)[-1][5:]
        print 'current:',detail

        # if 'busybox' in detail:
        #     continue

        # if 'openssl' in detail:
        #     continue
        if 'coreutils' in detail:
            if '6.12'  not in detail:
                continue
        # if 'openssl' and '1.0.1u' in detail:
        #     if 'O0-O3'  not in detail:
        #         continue
        # else:
        #     continue

            # if 'O1-O2'not in detail:
            #     continue

        # if 'openssl' in detail:
        #     if '1.0.1u' not in detail:
        #         continue

        train_file = config.DATASET_DIR + os.sep + "train_" + detail
        test_file = config.DATASET_DIR + os.sep + "test_" + detail
        valid_file = config.DATASET_DIR + os.sep + "vaild_" + detail

        TRAIN_TFRECORD = config.ROOT_DIR+ os.sep + "4_EmbeddingTFRecord" + os.sep + "jxd_debug"+ os.sep + "train_" + detail[:-6] + ".tfrecord"
        TEST_TFRECORD = config.ROOT_DIR+ os.sep + "4_EmbeddingTFRecord" + os.sep + "jxd_debug" + os.sep + "test_" + detail[:-6] + ".tfrecord"
        VALID_TFRECORD = config.ROOT_DIR+ os.sep + "4_EmbeddingTFRecord" + os.sep + "jxd_debug" + os.sep + "valid_" + detail[:-6] + ".tfrecord"
        # TRAIN_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "train_" + detail[:-6] + ".tfrecord"
        # TEST_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "test_" + detail[:-6] + ".tfrecord"
        # VALID_TFRECORD = config.TFRECORD_EMBEDDING_DIR + os.sep + "valid_" + detail[:-6] + ".tfrecord"


        train_pair, train_label, valid_pair, valid_label, test_pair, test_label = load_dataset()


        if not os.path.exists(TRAIN_TFRECORD):
        # if True:

            train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1, train_node_fea_2, train_num1, train_num2, train_max,\
                train_inst_1, train_inst_2, train_instnum_1, train_instnum_2 ,train_w2v_1,train_w2v_2,train_w2vnum_1,train_w2vnum_2,\
                = construct_learning_dataset(train_pair)
            # train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1, train_node_fea_2, train_num1, train_num2, train_max, \
            #     = construct_learning_dataset(train_pair)
            node_list = np.linspace(train_max, train_max, len(train_label), dtype=int)

            writer = tf.io.TFRecordWriter(TRAIN_TFRECORD)
            for item1, item2, item3, item4, item5, item6, item7, item8, item9, item10, item11, item12, item13, item14,item15,item16,item17,item18\
                    in itertools.izip(
                    train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1, train_node_fea_2,
                    train_num1, train_num2, node_list, train_inst_1, train_inst_2, train_instnum_1, train_instnum_2,\
                    train_w2v_1,train_w2v_2,train_w2vnum_1,train_w2vnum_2):
            # for item1, item2, item3, item4, item5, item6, item7, item8, item9, item10 in itertools.izip(
            #           train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1,
            #           train_node_fea_2, train_num1, train_num2, node_list):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[item1])),
                            'cfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                            'cfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                            'func_fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                            'func_fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                            'node_fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item6])),
                            'node_fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item7])),
                            'num1': tf.train.Feature(int64_list=tf.train.Int64List(value=[item8])),
                            'num2': tf.train.Feature(int64_list=tf.train.Int64List(value=[item9])),
                            'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item10])),
                            'inst_embedding1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item11])),
                            'inst_embedding2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item12])),
                            'inst_num1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item13])),
                            'inst_num2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item14])),
                            'w2v_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item15])),
                            'w2v_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item16])),
                            'w2v_num_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item17])),
                            'w2v_num_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item18])),
                        }))

                serialized = example.SerializeToString()
                writer.write(serialized)
            writer.close()

        if not os.path.exists(VALID_TFRECORD):
            valid_cfg_1, valid_cfg_2, valid_func_fea_1, valid_func_fea_2, valid_node_fea_1, valid_node_fea_2, valid_num1, valid_num2, valid_max, \
                valid_inst_1, valid_inst_2, valid_instnum_1, valid_instnum_2 ,valid_w2v_1,valid_w2v_2,valid_w2vnum_1,valid_w2vnum_2,\
                = construct_learning_dataset(valid_pair)
            node_list = np.linspace(valid_max, valid_max, len(valid_label), dtype=int)

            writer = tf.python_io.TFRecordWriter(VALID_TFRECORD)
            # for item1, item2, item3, item4, item5, item6, item7, item8, item9, item10, item11, item12, item13, item14 in itertools.izip(
            #         train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1, train_node_fea_2,
            #         train_num1, train_num2, node_list, train_inst_1, train_inst_2, train_instnum_1, train_instnum_2):
            for item1, item2, item3, item4, item5, item6, item7, item8, item9, item10 ,item11, item12, item13, item14,item15,item16,item17,item18\
                    in itertools.izip(
                    valid_label, valid_cfg_1, valid_cfg_2, valid_func_fea_1, valid_func_fea_2, valid_node_fea_1,
                    valid_node_fea_2, valid_num1, valid_num2, node_list,valid_inst_1, valid_inst_2, valid_instnum_1, valid_instnum_2,\
                    valid_w2v_1,valid_w2v_2,valid_w2vnum_1,valid_w2vnum_2,):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[item1])),
                            'cfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                            'cfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                            'func_fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                            'func_fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                            'node_fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item6])),
                            'node_fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item7])),
                            'num1': tf.train.Feature(int64_list=tf.train.Int64List(value=[item8])),
                            'num2': tf.train.Feature(int64_list=tf.train.Int64List(value=[item9])),
                            'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item10])),
                            'inst_embedding1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item11])),
                            'inst_embedding2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item12])),
                            'inst_num1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item13])),
                            'inst_num2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item14])),
                            'w2v_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item15])),
                            'w2v_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item16])),
                            'w2v_num_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item17])),
                            'w2v_num_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item18])),
                        }))

                serialized = example.SerializeToString()
                writer.write(serialized)
            writer.close()
        if not os.path.exists(TEST_TFRECORD):
            test_cfg_1, test_cfg_2, test_func_fea_1, test_func_fea_2, test_node_fea_1, test_node_fea_2, test_num1, test_num2, test_max, \
                test_inst_1, test_inst_2, test_instnum_1, test_instnum_2 ,test_w2v_1,test_w2v_2,test_w2vnum_1,test_w2vnum_2,\
                = construct_learning_dataset(test_pair)
            node_list = np.linspace(test_max, test_max, len(test_label), dtype=int)

            writer = tf.python_io.TFRecordWriter(TEST_TFRECORD)
            # for item1, item2, item3, item4, item5, item6, item7, item8, item9, item10, item11, item12, item13, item14 in itertools.izip(
            #         train_label, train_cfg_1, train_cfg_2, train_func_fea_1, train_func_fea_2, train_node_fea_1, train_node_fea_2,
            #         train_num1, train_num2, node_list, train_inst_1, train_inst_2, train_instnum_1, train_instnum_2):
            for item1, item2, item3, item4, item5, item6, item7, item8, item9, item10 ,item11, item12, item13, item14 , item15,item16,item17,item18 \
                    in itertools.izip(
                    test_label, test_cfg_1, test_cfg_2, test_func_fea_1, test_func_fea_2, test_node_fea_1,
                    test_node_fea_2, test_num1, test_num2, node_list,test_inst_1, test_inst_2, test_instnum_1, test_instnum_2,\
                    test_w2v_1,test_w2v_2,test_w2vnum_1,test_w2vnum_2,):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[item1])),
                            'cfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                            'cfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                            'func_fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                            'func_fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                            'node_fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item6])),
                            'node_fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item7])),
                            'num1': tf.train.Feature(int64_list=tf.train.Int64List(value=[item8])),
                            'num2': tf.train.Feature(int64_list=tf.train.Int64List(value=[item9])),
                            'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item10])),
                            'inst_embedding1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item11])),
                            'inst_embedding2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item12])),
                            'inst_num1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item13])),
                            'inst_num2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item14])),
                            'w2v_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item15])),
                            'w2v_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item16])),
                            'w2v_num_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item17])),
                            'w2v_num_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item18])),
                        }))
                serialized = example.SerializeToString()
                writer.write(serialized)
            writer.close()


