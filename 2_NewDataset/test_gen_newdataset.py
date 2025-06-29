#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import csv
import os
import glob
import random

import json
import numpy as np

# a = [29,36,57,12,79,43,23,56,28,11,14,15,16,37,24,35,17,24,33,15,39,46,52,13]
# b = dict(Counter(a))
# print ([key for key,value in b.items()if value > 1])  #只展示重复元素
# print ({key:value for key,value in b.items()if value > 1})  #展现重复元素和重复次数
# from six.moves import xrange
## id,func_name,func_id,block_id_in_func,numeric constants,string constants,No. of transfer instructions,No. of calls,No. of instructinos,No. of arithmetic instructions,No. of logic instructions,No. of offspring,betweenness centrality
#生成训练集中函数block的数量
# block_num_min< block_num <= block_num_max
# if block_num_max = -1, 忽略这一设置,不考虑block数量
# block_num_min = 2
# block_num_max = 30
block_num_max = -1
block_num_min = -1



#正例负例数量
pos_num = 5
neg_num = 5


json_path = '../6_json_files'

subset = [0.8,0.1,0.1]



# for json_list in os.listdir(json_path):
#     filePath = os.path.join(json_path, json_list)
#     json_name = json_list.split('_')
#     funlist_name = str(json_name[0]+"_"+json_name[1]+"_"+json_name[2]+"_"+json_name[3])
#     funcPath = os.path.join(func_list_path, funlist_name+"_fea.csv")
#     print filePath
#     print funcPath
#     with open(filePath, 'r') as f:
#         func_name = []
#         for line1 in f.readlines():
#             data = json.loads(line1)
#             func_name.append(data['fname'])
#
#     with open(funcPath, 'w') as f:
#
#         '''
#         删除列表中重复元素
#         '''
#         b = dict(Counter(func_name))
#         x = ([key for key, value in b.items() if value > 1])  # 只展示重复元素
#         y = ({key: value for key, value in b.items() if value > 1})
#
#         for q in x:
#             num = y[q]
#             for i in range(num):
#                 func_name.remove(q)
#
#         for line in func_name:
#             function_str = str(line)+","+str(json_list)+"\n"
#             f.write(function_str)


def get_f_dict(json_list):
    fname_num = 0
    fname_dict = {}
    fname_list = []

    for f_name in json_list:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if g_info['n_num'] < 5:
                    continue
                if g_info['fname'] not in fname_dict:
                    fname_dict[g_info['fname']] = []
                    fname_list.append(g_info['fname'])
                    fname_dict[g_info['fname']].append(f_name)
                    fname_num += 1
                else:
                    fname_dict[g_info['fname']].append(f_name)


    return fname_dict,fname_list


def partition_data(Gs, classes, partitions, perm):
    C = len(classes)  # 函数个数
    st = 0.0  # start开始的函数位置
    ret = []
    for part in partitions:
        cur_g = []
        cur_c = []
        ed = st + part * C  # part*C是函数个数，end也就是结束的函数位置
        for cls in range(int(st), int(ed)):
            prev_class = classes[perm[cls]]  # 随机取样出一个函数，获得它所有的图
            cur_c.append([])
            for i in range(len(prev_class)):  # 取出该函数的所有图，把图存到cur_g里
                cur_g.append(Gs[prev_class[i]])
                cur_g[-1].label = len(cur_c)-1
                cur_c[-1].append(len(cur_g)-1)

        ret.append(cur_g)  # cur_g是这一阶段的所有函数的所有图的一个列表
        ret.append(cur_c)  # cur_c是函数个数大小的一个列表，每个元素是一个列表，对应一个函数的图的位置
        st = ed

    return ret  # ret长度为6，一个train图的列表，一个train的label列表，一个dev图的列表，一个dev的label列表，一个test图的列表，一个test的label列表



def func_split(func_list,partitions,perm):

    fnum = len(func_dict)
    st = 0.0
    ret = []

    for part in partitions:
        cur_f = []
        ed = st + part * fnum

        for index in range(int(st),int(ed)):
            fname = func_list[perm[index]]
            cur_f.append(fname)

        st = ed
        ret.append(cur_f)

    return ret



def generate_pair(func_list, func_dict, fp):

    posfuncpair_dict = {}
    negfuncpari_dict = {}

    pos_list = []
    neg_list = []

    fpos_list = []
    fneg_list = []

    maxn = -1

    newfunc_list = []

    for func in func_list:

        source_files = func_dict[func]
        if len(source_files) < 2:
            continue
        if len(source_files) > maxn:
            maxn = len(source_files)

        for s0 in source_files:
            s1 = random.sample(source_files,1)
            while(s0==s1):
                s1 = random.sample(source_files,1)

            pos_str = str(s0) + "," + str(func) + "," + str(s1[0]) + "," + str(func) + ",1\n"
            fpos_list.append(pos_str)

            func2 = random.sample(func_list, 1)
            while (func2 == func):
                func2 = random.sample(func_list, 1)

            s2 = random.sample(func_dict[func2[0]], 1)
            neg_str = str(s0) + "," + str(func) + "," + str(s2[0]) + "," + str(func2[0]) + ",-1\n"
            fneg_list.append(neg_str)


        posfuncpair_dict[func] = fpos_list
        negfuncpari_dict[func] = fneg_list
        fpos_list = []
        fneg_list = []
        newfunc_list.append(func)





    for i in range(maxn):
        for func in newfunc_list:
            if len(negfuncpari_dict[func]) < i+1:
                continue

            fnlist = negfuncpari_dict[func]
            fplist = posfuncpair_dict[func]

            pos_list.append(fplist[i])
            neg_list.append(fnlist[i])

    #
    # random.shuffle(neg_list)
    # random.shuffle(pos_list)
    neg_list = list(reversed(neg_list))
    f1 = []
    f2 = []

    for i in range(len(pos_list)):
        f1.append(pos_list[i])
        f2.append(neg_list[i])

        if (i+1) % 5==0 and i != 0:
            for pos in f1:
                fp.write(pos)
            for neg in f2:
                fp.write(neg)
            f1 = []
            f2 = []


    return 1




with open('./dataset_list.csv','r') as fp:
    programs = csv.reader(fp)

    for program in programs:

        program_name = program[0]
        program_version = program[1]
        program_arch = program[2]
        program_opti = program[3]

        PREFIX = program_name + '_' + program_version + '_' + program_arch + '_' + program_opti

        func_file = './func_list'

        train_file = func_file + os.sep + "train_" + PREFIX + "_t.csv"
        test_file = func_file + os.sep + "test_" + PREFIX + "_t.csv"
        vaild_file = func_file + os.sep + "vaild_" + PREFIX + "_t.csv"

        json_list = []
        jname_split = []

        program_arch_list = program_arch.split('-')
        program_opti_list = program_opti.split('-')

        program_nv = (program_name,program_version)


        for json_name in os.listdir(json_path):

            json_name_list = json_name.split('_')

            json_name_nv = (json_name_list[0], json_name_list[1])

            if json_name_nv == program_nv:

                if json_name_list[2] in program_arch_list and json_name_list[3] in program_opti_list:

                    filepath = os.path.join(json_path, json_name)
                    json_list.append(filepath)


        func_dict = {}
        func_list = []

        func_dict,func_list = get_f_dict(json_list)

        perm = np.random.permutation(len(func_list))

        ret = func_split(func_list,subset,perm)

        func_list_train = ret[0]
        func_list_vaild = ret[1]
        func_list_test = ret[2]

        train_fp = open(train_file, "w")
        test_fp = open(test_file, "w")
        vaild_fp = open(vaild_file, "w")


        a = generate_pair(func_list_train, func_dict, train_fp)
        b = generate_pair(func_list_vaild, func_dict, vaild_fp)
        c = generate_pair(func_list_test, func_dict, test_fp)








