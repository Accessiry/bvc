#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os


#program_name = "coreutils"
#program_version = "6.12"
#program_arch = "arm"
#program_opti = "O2-O3"

program_name = "coreutils"
program_version = "6.12"
program_arch = "arm_mips"
program_opti = "O1"






# IDA Path

IDA32_DIR = "/home/zrf/Desktop/project/ida2/ida2/idaq"
IDA64_DIR = "/home/zrf/Desktop/project/ida2/ida2/idaq64"

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) #  The path of the current file

# # 傅滢的！！！
# IDA32_DIR = "D:\\IDA7.0\\ida.exe"
# IDA64_DIR = "D:\\IDA7.0\\ida64.exe"
#
# ROOT_DIR = "D:\\IDA7.0\\python"#  The path of the current file
#!/usr/bin/python
# -*- coding: UTF-8 -*-

ZTHtest = ROOT_DIR+os.sep+"csv_result"

CODE_DIR = ROOT_DIR
O_DIR = ROOT_DIR+ os.sep + "0_Libs"                  #  The root path of All the binary file
IDB_DIR = ROOT_DIR+ os.sep + "0_Libs"                #  The root path of All the idb file
FEA_DIR = ROOT_DIR+ os.sep + "1_Features"            #  The root path of  the feature file

DATASET_DIR = ROOT_DIR+ os.sep + "2_NewDataset" + os.sep + 'func_list'
DATASET_DIR_libgmp_libsqlite = ROOT_DIR+ os.sep + 'Dataset_zzz'
# DATASET_DIR = ROOT_DIR+ os.sep + "test_Dataset"

# original result
VULSEEKER_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "jxd" + os.sep + "vulseeker"
SENSE_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "jxd" + os.sep + "sense"
FIT_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "jxd" + os.sep + "fit"
SAFE_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "jxd" + os.sep + "safe"
BERT_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "jxd" + os.sep + "bert"

# bert result
BERT_VULSEEKER_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "Bert" + os.sep + "VulSeeker"
BERT_SENSE_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "Bert" + os.sep + "SENSE"
BERT_FIT_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "Bert" + os.sep + "FIT"
BERT_SAFE_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "Bert" + os.sep + "SAFE"

# W2V result
W2V_VUL_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "Word2Vec" + os.sep + "VulSeeker"
W2V_SEN_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "Word2Vec" + os.sep + "SENSE"
W2V_FIT_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "Word2Vec" + os.sep + "FIT"
W2V_SAFE_RESULT_DIR = ROOT_DIR+ os.sep + "Experimental results" + os.sep + "Word2Vec" + os.sep + "SAFE"

PIC_DIR = ROOT_DIR+ os.sep + "picture" + os.sep + program_name
INSTR_PIC_DIR = ROOT_DIR+ os.sep + "picture_instr" + os.sep + program_name

TFRECORD_GEMINI_DIR = ROOT_DIR+ os.sep + "3_TFRecord"+  os.sep + "Gemini"


# TFRecord path
TFRECORD_VULSEEKER_DIR = ROOT_DIR+ os.sep + "3_TFRecord" + os.sep + "VulSeeker"
TFRECORD_EMBEDDING_DIR = ROOT_DIR+ os.sep + "4_EmbeddingTFRecord" + os.sep + "w2v_bert"
# TFRECORD_EMBEDDING_DIR_debug = ROOT_DIR+ os.sep + "4_EmbeddingTFRecord" + os.sep + "jxd_debug"
TFRECORD_EMBEDDING_DIR_OLD = '/media/ubuntu/Seagate Backup Plus Drive/w2v_bert'
TFRECORD_EMBEDDING_DIR_libgmp_libsqlite = '/media/ubuntu/Seagate Backup Plus Drive/libgmp_libsqlite_tfrecord'

#libgmp_libsqlite-bert-path
Libgmp_Libsqlite_bert_path = '/media/ubuntu/Seagate Backup Plus Drive/libgmp_libsqlite3_bert'
Libgmp_Libsqlite_w2v_path = '/media/ubuntu/Seagate Backup Plus Drive/libgmp-libsqlite_w2v'


MODEL_GEMINI_DIR = ROOT_DIR+ os.sep + "4_Model" + os.sep + "Gemini"
MODEL_VULSEEKER_DIR = ROOT_DIR+ os.sep + "4_Model" + os.sep + "VulSeeker"
CVE_FEATURE_DIR = ROOT_DIR+ os.sep + "5_CVE_Feature"
SEARCH_GEMINI_TFRECORD_DIR = ROOT_DIR+ os.sep + "6_Search_TFRecord" + os.sep + "Gemini"
SEARCH_VULSEEKER_TFRECORD_DIR = ROOT_DIR+ os.sep + "6_Search_TFRecord" + os.sep + "VulSeeker"
SEARCH_RESULT_GEMINI_DIR = ROOT_DIR+ os.sep + "7_Search_Result" + os.sep + "Gemini"
SEARCH_RESULT_VULSEEKER_DIR = ROOT_DIR+ os.sep + "7_Search_Result" + os.sep + "VulSeeker"
LOG_dir = ROOT_DIR + os.sep + "logg" + os.sep + "VulSeeker"


