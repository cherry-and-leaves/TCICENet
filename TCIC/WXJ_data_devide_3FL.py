# -*- coding:utf-8 -*-

import os
import sys
import time
import pickle
import random
import numpy as np
import _pickle as cPickle
class_num = 3
num = 64
image_size = 170
img_channels = 1

# ========================================================== #
# ├─ prepare_data()
#  ├─ download training data if not exist by download_data()
#  ├─ load data by load_data()
#  └─ shuffe and return data
# ========================================================== #
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_pkl(pkl_file):
    print("[INFO] loading images from pkl...")
    read_file_1 = open(pkl_file, 'rb')
    cPickle_image = cPickle.load(read_file_1)
    read_file_1.close()
    data = np.array(cPickle_image, dtype="float32")
    data2 = data[:, :, :, 0:1]
    # data = data.resize(data.shape[0], image_size, image_size, img_channels)
    # data = data
    # 归一化
    # amin, amax = data.min(), data.max()  # 求最大最小值
    # data = (data - amin) / (amax - amin)
    # data = data / 255.0
    return data2
def load_label_pkl(pkl_file):
    print("[INFO] loading label from pkl...")
    read_file_1 = open(pkl_file, 'rb')
    cPickle_label = cPickle.load(read_file_1)
    read_file_1.close()
    labels = np.array(cPickle_label, dtype="float")
    labels = to_categorical(labels, num_classes=class_num)
    return labels
def prepare_data():
    print("======Loading train data======")
    train_data_1 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\train_devide3_1-17641_140.pkl')
    train_labels_1 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\train_devide3_1-17641_label_140.pkl')
    train_data_2 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\train_devide3_17641-35282_140.pkl')
    train_labels_2 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\train_devide3_17641-35282_label_140.pkl')
    print("======Loading test data======")
    test_data = load_data_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\test_devide3_140.pkl')
    test_labels = load_label_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\test_devide3_label_140.pkl')
    print("======Load finished======")
    print("======Shuffling data======")
    indices_1 = np.random.permutation(len(train_data_1))
    train_data_1 = train_data_1[indices_1]
    train_labels_1 = train_labels_1[indices_1]
    indices_2 = np.random.permutation(len(train_data_2))
    train_data_2 = train_data_2[indices_2]
    train_labels_2 = train_labels_2[indices_2]
    print("=======Append data ========")
    train_data = np.append(train_data_1, train_data_2, axis=0)############加进来
    train_labels = np.append(train_labels_1, train_labels_2, axis=0)
    # train_data = train_data_1############不加
    # train_labels = train_labels_1
    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Prepare Finished======")
    return train_data, train_labels, test_data, test_labels


# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# ========================================================== #
def prepare_data_4_32_256():
    print("======Loading train data======")
    train_data_1 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_1-8820_256.pkl')
    train_labels_1 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_1-8820_256_label.pkl')
    print("Train data:", np.shape(train_data_1), np.shape(train_labels_1))
    train_data_2 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_8821-17640_256.pkl')
    train_labels_2 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_8821-17640_256_label.pkl')
    print("Train data:", np.shape(train_data_2), np.shape(train_labels_2))
    train_data_3 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_17641-26460_256.pkl')
    train_labels_3 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_17641-26460_256_label.pkl')
    print("Train data:", np.shape(train_data_3), np.shape(train_labels_3))
    train_data_4 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_26461-30461_256.pkl')
    train_labels_4 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_26461-30461_256_label.pkl')
    print("Train data:", np.shape(train_data_4), np.shape(train_labels_4))
    train_data_5 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_30462-35282_256.pkl')
    train_labels_5 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_30462-35282_256_label.pkl')
    print("Train data:", np.shape(train_data_5), np.shape(train_labels_5))
    print("======Loading test data======")
    test_data = load_data_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\2017-2019_256_256.pkl')
    test_labels = load_label_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\2017-2019_256_label.pkl')
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")
    print("======Shuffling data======")
    indices_1 = np.random.permutation(len(train_data_1))
    train_data_1 = train_data_1[indices_1]
    train_labels_1 = train_labels_1[indices_1]
    indices_2 = np.random.permutation(len(train_data_2))
    train_data_2 = train_data_2[indices_2]
    train_labels_2 = train_labels_2[indices_2]
    indices_3 = np.random.permutation(len(train_data_3))
    train_data_3 = train_data_3[indices_3]
    train_labels_3 = train_labels_3[indices_3]
    indices_4 = np.random.permutation(len(train_data_4))
    train_data_4 = train_data_4[indices_4]
    train_labels_4 = train_labels_4[indices_4]
    indices_5 = np.random.permutation(len(train_data_5))
    train_data_5 = train_data_5[indices_5]
    train_labels_5 = train_labels_5[indices_5]
    print("=======Append data ========")
    train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5), axis=0)############加进来
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5), axis=0)
    #     # train_data = train_data_1############不加
    # train_labels = train_labels_1
    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Prepare Finished======")
    return train_data, train_labels, test_data, test_labels

###############35282——170
def prepare_data_4_32_170():
    print("======Loading train data======")
    train_data_1 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_1-8820_170.pkl')
    train_labels_1 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_1-8820_170_label.pkl')
    print("Train data:", np.shape(train_data_1), np.shape(train_labels_1))
    train_data_2 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_8821-17640_170.pkl')
    train_labels_2 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_8821-17640_170_label.pkl')
    print("Train data:", np.shape(train_data_2), np.shape(train_labels_2))
    train_data_3 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_17641-26460_170.pkl')
    train_labels_3 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_17641-26460_170_label.pkl')
    print("Train data:", np.shape(train_data_3), np.shape(train_labels_3))
    train_data_4 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_26461-30461_170.pkl')
    train_labels_4 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_26461-30461_170_label.pkl')
    print("Train data:", np.shape(train_data_4), np.shape(train_labels_4))
    train_data_5 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_30462-35282_170.pkl')
    train_labels_5 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl\train_devide3_30462-35282_170_label.pkl')
    print("Train data:", np.shape(train_data_5), np.shape(train_labels_5))
    print("======Loading test data======")
    test_data = load_data_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\2017-2019_170.pkl')
    test_labels = load_label_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\2017-2019_170_label.pkl')
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")
    print("======Shuffling data======")
    indices_1 = np.random.permutation(len(train_data_1))
    train_data_1 = train_data_1[indices_1]
    train_labels_1 = train_labels_1[indices_1]
    indices_2 = np.random.permutation(len(train_data_2))
    train_data_2 = train_data_2[indices_2]
    train_labels_2 = train_labels_2[indices_2]
    indices_3 = np.random.permutation(len(train_data_3))
    train_data_3 = train_data_3[indices_3]
    train_labels_3 = train_labels_3[indices_3]
    indices_4 = np.random.permutation(len(train_data_4))
    train_data_4 = train_data_4[indices_4]
    train_labels_4 = train_labels_4[indices_4]
    indices_5 = np.random.permutation(len(train_data_5))
    train_data_5 = train_data_5[indices_5]
    train_labels_5 = train_labels_5[indices_5]
    print("=======Append data ========")
    train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5), axis=0)############加进来
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5), axis=0)
    #     # train_data = train_data_1############不加
    # train_labels = train_labels_1
    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Prepare Finished======")
    return train_data, train_labels, test_data, test_labels



#####################6wan
def prepare_data_6wan():
    print("======Loading train data======")
    train_data_1 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\1-8000_256.pkl')
    train_labels_1 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\1-8000_256_label.pkl')
    print("Train data:", np.shape(train_data_1), np.shape(train_labels_1))
    train_data_2 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\8001-16000_256.pkl')
    train_labels_2 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\8001-16000_256_label.pkl')
    print("Train data:", np.shape(train_data_2), np.shape(train_labels_2))
    train_data_3 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\16001-24000_256.pkl')
    train_labels_3 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\16001-24000_256_label.pkl')
    print("Train data:", np.shape(train_data_3), np.shape(train_labels_3))
    train_data_4 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\24001-32000_256.pkl')
    train_labels_4 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\24001-32000_256_label.pkl')
    print("Train data:", np.shape(train_data_4), np.shape(train_labels_4))
    train_data_5 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\32001-40000_256.pkl')
    train_labels_5 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\32001-40000_256_label.pkl')
    print("Train data:", np.shape(train_data_5), np.shape(train_labels_5))
    train_data_6 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\40001-48000_256.pkl')
    train_labels_6 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\40001-48000_256_label.pkl')
    print("Train data:", np.shape(train_data_6), np.shape(train_labels_6))
    train_data_7 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\48001-56000_256.pkl')
    train_labels_7 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\48001-56000_256_label.pkl')
    print("Train data:", np.shape(train_data_7), np.shape(train_labels_7))
    train_data_8 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\56001-62396_256.pkl')
    train_labels_8 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\56001-62396_256_label.pkl')
    print("Train data:", np.shape(train_data_8), np.shape(train_labels_8))
    print("======Loading test data======")
    test_data = load_data_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\2017-2019_256_256.pkl')
    test_labels = load_label_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\2017-2019_256_label.pkl')
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")
    print("======Shuffling data======")
    indices_1 = np.random.permutation(len(train_data_1))
    train_data_1 = train_data_1[indices_1]
    train_labels_1 = train_labels_1[indices_1]
    indices_2 = np.random.permutation(len(train_data_2))
    train_data_2 = train_data_2[indices_2]
    train_labels_2 = train_labels_2[indices_2]
    indices_3 = np.random.permutation(len(train_data_3))
    train_data_3 = train_data_3[indices_3]
    train_labels_3 = train_labels_3[indices_3]
    indices_4 = np.random.permutation(len(train_data_4))
    train_data_4 = train_data_4[indices_4]
    train_labels_4 = train_labels_4[indices_4]
    indices_5 = np.random.permutation(len(train_data_5))
    train_data_5 = train_data_5[indices_5]
    train_labels_5 = train_labels_5[indices_5]
    indices_6 = np.random.permutation(len(train_data_6))
    train_data_6 = train_data_6[indices_6]
    train_labels_6 = train_labels_6[indices_6]
    indices_7 = np.random.permutation(len(train_data_7))
    train_data_7 = train_data_7[indices_7]
    train_labels_7 = train_labels_7[indices_7]
    indices_8 = np.random.permutation(len(train_data_8))
    train_data_8 = train_data_8[indices_8]
    train_labels_8 = train_labels_8[indices_8]
    print("=======Append data ========")
    train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5
                                 , train_data_6, train_data_7, train_data_8), axis=0)############加进来
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5
                                   , train_labels_6, train_labels_7, train_labels_8), axis=0)
    #     # train_data = train_data_1############不加
    # train_labels = train_labels_1
    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Prepare Finished======")
    return train_data, train_labels, test_data, test_labels

#####################6wan_shanchu
def prepare_data_6wan_shanchu():
    print("======Loading train data======")
    train_data_1 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\shanchu_1-8000_256.pkl')
    train_labels_1 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\shanchu_1-8000_256_label.pkl')
    print("Train data:", np.shape(train_data_1), np.shape(train_labels_1))
    train_data_2 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\8001-16000_256.pkl')
    train_labels_2 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\8001-16000_256_label.pkl')
    print("Train data:", np.shape(train_data_2), np.shape(train_labels_2))
    train_data_3 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\shanchu_16001-24000_256.pkl')
    train_labels_3 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\shanchu_16001-24000_256_label.pkl')
    print("Train data:", np.shape(train_data_3), np.shape(train_labels_3))
    train_data_4 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\shanchu_24001-32000_256.pkl')
    train_labels_4 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\shanchu_24001-32000_256_label.pkl')
    print("Train data:", np.shape(train_data_4), np.shape(train_labels_4))
    train_data_5 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\32001-40000_256.pkl')
    train_labels_5 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\32001-40000_256_label.pkl')
    print("Train data:", np.shape(train_data_5), np.shape(train_labels_5))
    train_data_6 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\40001-48000_256.pkl')
    train_labels_6 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\40001-48000_256_label.pkl')
    print("Train data:", np.shape(train_data_6), np.shape(train_labels_6))
    train_data_7 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\48001-56000_256.pkl')
    train_labels_7 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\48001-56000_256_label.pkl')
    print("Train data:", np.shape(train_data_7), np.shape(train_labels_7))
    train_data_8 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\56001-62396_256.pkl')
    train_labels_8 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\pkl_6wan\56001-62396_256_label.pkl')
    print("Train data:", np.shape(train_data_8), np.shape(train_labels_8))
    print("======Loading test data======")
    test_data = load_data_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\2017-2019_256_256.pkl')
    test_labels = load_label_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\2017-2019_256_label.pkl')
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")
    print("======Shuffling data======")
    indices_1 = np.random.permutation(len(train_data_1))
    train_data_1 = train_data_1[indices_1]
    train_labels_1 = train_labels_1[indices_1]
    indices_2 = np.random.permutation(len(train_data_2))
    train_data_2 = train_data_2[indices_2]
    train_labels_2 = train_labels_2[indices_2]
    indices_3 = np.random.permutation(len(train_data_3))
    train_data_3 = train_data_3[indices_3]
    train_labels_3 = train_labels_3[indices_3]
    indices_4 = np.random.permutation(len(train_data_4))
    train_data_4 = train_data_4[indices_4]
    train_labels_4 = train_labels_4[indices_4]
    indices_5 = np.random.permutation(len(train_data_5))
    train_data_5 = train_data_5[indices_5]
    train_labels_5 = train_labels_5[indices_5]
    indices_6 = np.random.permutation(len(train_data_6))
    train_data_6 = train_data_6[indices_6]
    train_labels_6 = train_labels_6[indices_6]
    indices_7 = np.random.permutation(len(train_data_7))
    train_data_7 = train_data_7[indices_7]
    train_labels_7 = train_labels_7[indices_7]
    indices_8 = np.random.permutation(len(train_data_8))
    train_data_8 = train_data_8[indices_8]
    train_labels_8 = train_labels_8[indices_8]
    print("=======Append data ========")
    train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5
                                 , train_data_6, train_data_7, train_data_8), axis=0)############加进来
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5
                                   , train_labels_6, train_labels_7, train_labels_8), axis=0)
    #     # train_data = train_data_1############不加
    # train_labels = train_labels_1
    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Prepare Finished======")
    return train_data, train_labels, test_data, test_labels


def prepare_data_4():
    print("======Loading train data======")
    train_data_1 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_1-8820_256.pkl')
    train_labels_1 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_1-8820_256_label.pkl')
    print("Train data:", np.shape(train_data_1), np.shape(train_labels_1))
    train_data_2 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_8821-17640_256.pkl')
    train_labels_2 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_8821-17640_256_label.pkl')
    print("Train data:", np.shape(train_data_2), np.shape(train_labels_2))
    train_data_3 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_17641-26460_256.pkl')
    train_labels_3 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_17641-26460_256_label.pkl')
    print("Train data:", np.shape(train_data_3), np.shape(train_labels_3))
    train_data_4 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_26461-30461_256.pkl')
    train_labels_4 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_26461-30461_256_label.pkl')
    print("Train data:", np.shape(train_data_4), np.shape(train_labels_4))
    train_data_5 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_30462-35282_256.pkl')
    train_labels_5 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_30462-35282_256_label.pkl')
    print("Train data:", np.shape(train_data_5), np.shape(train_labels_5))
    print("======Loading test data======")
    test_data = load_data_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\test_devide3_256.pkl')
    test_labels = load_label_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\test_devide3_label_256.pkl')
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")
    print("======Shuffling data======")
    indices_1 = np.random.permutation(len(train_data_1))
    train_data_1 = train_data_1[indices_1]
    train_labels_1 = train_labels_1[indices_1]
    indices_2 = np.random.permutation(len(train_data_2))
    train_data_2 = train_data_2[indices_2]
    train_labels_2 = train_labels_2[indices_2]
    indices_3 = np.random.permutation(len(train_data_3))
    train_data_3 = train_data_3[indices_3]
    train_labels_3 = train_labels_3[indices_3]
    indices_4 = np.random.permutation(len(train_data_4))
    train_data_4 = train_data_4[indices_4]
    train_labels_4 = train_labels_4[indices_4]
    indices_5 = np.random.permutation(len(train_data_5))
    train_data_5 = train_data_5[indices_5]
    train_labels_5 = train_labels_5[indices_5]
    # print("=======Append data ========")
    train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5), axis=0)############加进来
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5), axis=0)
    # train_data = train_data_1############不加
    # train_labels = train_labels_1
    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Prepare Finished======")
    return train_data, train_labels, test_data, test_labels

#########################WXJ20200115
def test():

    test_data = load_data_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\tys_2014-2019_256.pkl')
    test_labels = load_label_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\tys_2014-2019_256_label.pkl')
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Prepare Finished======")
    return test_data, test_labels

def prepare_data_4():
    print("======Loading train data======")
    train_data_1 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_1-8820_256.pkl')
    train_labels_1 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_1-8820_256_label.pkl')
    print("Train data:", np.shape(train_data_1), np.shape(train_labels_1))
    # train_data_2 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_8821-17640_256.pkl')
    # train_labels_2 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_8821-17640_256_label.pkl')
    # print("Train data:", np.shape(train_data_2), np.shape(train_labels_2))
    # train_data_3 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_17641-26460_256.pkl')
    # train_labels_3 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_17641-26460_256_label.pkl')
    # print("Train data:", np.shape(train_data_3), np.shape(train_labels_3))
    # train_data_4 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_26461-30461_256.pkl')
    # train_labels_4 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_26461-30461_256_label.pkl')
    # print("Train data:", np.shape(train_data_4), np.shape(train_labels_4))
    # train_data_5 = load_data_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_30462-35282_256.pkl')
    # train_labels_5 = load_label_pkl(r'F:\1xjie\CLASS_Japan\DEVIDE_RAW_256\4_pkl\train_devide3_30462-35282_256_label.pkl')
    # print("Train data:", np.shape(train_data_5), np.shape(train_labels_5))
    print("======Loading test data======")
    test_data = load_data_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\test_devide3_256.pkl')
    test_labels = load_label_pkl(r'F:\1xjie\CLASS_Japan\deide_3_predict_256\test_devide3_label_256.pkl')
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")
    print("======Shuffling data======")
    indices_1 = np.random.permutation(len(train_data_1))
    train_data_1 = train_data_1[indices_1]
    train_labels_1 = train_labels_1[indices_1]
    # indices_2 = np.random.permutation(len(train_data_2))
    # train_data_2 = train_data_2[indices_2]
    # train_labels_2 = train_labels_2[indices_2]
    # indices_3 = np.random.permutation(len(train_data_3))
    # train_data_3 = train_data_3[indices_3]
    # train_labels_3 = train_labels_3[indices_3]
    # indices_4 = np.random.permutation(len(train_data_4))
    # train_data_4 = train_data_4[indices_4]
    # train_labels_4 = train_labels_4[indices_4]
    # indices_5 = np.random.permutation(len(train_data_5))
    # train_data_5 = train_data_5[indices_5]
    # train_labels_5 = train_labels_5[indices_5]
    print("=======Append data ========")
    # train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5), axis=0)############加进来
    # train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5), axis=0)
    train_data = train_data_1############不加
    train_labels = train_labels_1
    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Prepare Finished======")
    return train_data, train_labels, test_data, test_labels
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):

    # batch = _random_flip_leftright(batch)
    # batch = _random_crop(batch, [image_size, image_size], 4)
    return batch