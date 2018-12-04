#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

# 文件说明：利用多GPU训练模型

from __future__ import print_function
from keras.models import *
from keras.layers import Input, Lambda
from keras import backend as K
from keras.layers.merge import concatenate
import tensorflow as tf
import numpy as np
import os


# os.environ['CUDA_VISIBLE_DEVICES'] = gpus  # =>availabel devices: 0,1,2,3,4,5
# import tensorflow as tf
# session_config=tf.ConfigProto()
# #session_config.gpu_options.allow_growth = True
# session_config.gpu_options.per_process_gpu_memory_fraction=ratio
# session_config.allow_soft_placement = True
# session = tf.Session(config=session_config)

USE_FREE_MEMORY_RATIO=1.0 #使用可用内存的多少比例

# 自动获取可用gpu
# input_ngpu: int类型, 表示需要的gpu数量(为None时表示使用全部可用gpu)
# free_ratio: float类型, 表示选择的gpu必须满足可用内存占比,默认有50%以上的可用内存才是可用gpu
# exclude_gpus: 需要排除的gpus
# 使用方法: 设置需要使用的gpu数量, 需要的gpu空闲内存所占的比例; 可设置free_ration,这样只选择没人用的gpu
# eg: gpus, ratio=get_available_gpus(input_ngpu=4, free_ratio=0.5, orderby_free_memory=True)
def get_available_gpus(input_ngpu=None, free_ratio=0.5, orderby_free_memory=False, exclude_gpus=[]):
    params = os.popen("nvidia-smi --query-gpu=index,pci.bus_id,gpu_name,memory.used,memory.free,memory.total,power.draw,power.limit --format=csv ")  # > gpu1.txt
    gpus_info = params.readlines()  # 不能重复读,第一行是列名

    available=[]
    available_used=[]   #可用gpu按free memory排序,然后选取最空闲的gpu
    for index in range(1, len(gpus_info)):
        gpu_info = gpus_info[index].split(",")
        gpu_id = gpu_info[0]
        memory_free = float(gpu_info[4].strip().split(" ")[0])
        memory_total = float(gpu_info[5].strip().split(" ")[0])
        memory_free_ratio = memory_free / memory_total
        if memory_free_ratio>=free_ratio-0.05:  #有可能显存会轻微地波动
                available_used.append(memory_free_ratio)
                available.append(int(gpu_id))
                # min_memory_free_ratio=min(min_memory_free_ratio, memory_free_ratio)

    if input_ngpu is not None:
        if len(available)<input_ngpu:
            gpus_num=len(available)
            print("you want to use %d gpus, but only %d gpus available." %(input_ngpu, len(available)))
        else:
            gpus_num=input_ngpu

    # 按名称对可用gpu排序
    available=[str(val) for val in available]
    available.sort()

    # 排除指定的不可用的gpus
    if len(exclude_gpus) > 0:
        available_used = [available_used[idx] for idx in range(len(available)) if available[idx] not in exclude_gpus]
        available = [val for val in available if val not in exclude_gpus]

    all_gpus=",".join(available)

    if orderby_free_memory:
        available = np.array(available)[np.array(available_used).argsort()[::-1]]  # np.argsort只能返回按从小到大排的索引
    gpus=",".join(sorted(available[:gpus_num]))

    if gpus_num:
        print("all available gpus: ", all_gpus)
        print("you will use gpus: ", gpus)
    return gpus, (free_ratio-0.05)*USE_FREE_MEMORY_RATIO    #返回原gpu可用百分比*0.6为GPU内存使用比


# 人为指定使用那几块GPU
# eg: hand_set_gpus(gpus="0,1,2", memory_ratio=0.8, auto_growth=False)
def hand_set_gpus(gpus="", memory_ratio=0.8, auto_growth=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus  # =>availabel devices: 0,1,2,3,4,5

    session_config = tf.ConfigProto()
    if auto_growth:
        session_config.gpu_options.allow_growth = True
    else:
        session_config.gpu_options.per_process_gpu_memory_fraction = memory_ratio
    session_config.allow_soft_placement = True
    session = tf.Session(config=session_config)
    return session


# 智能化指定可用GPU(独立成函数,避免出现覆盖的情况)
#eg: set_gpus(num_gpus=2, free_ratio=1.0, auto_growth=False)
def set_gpus(num_gpus, free_ratio=1.0, auto_growth=False, exclude_gpus=[]):
    gpus, ratio = get_available_gpus(input_ngpu=num_gpus, free_ratio=free_ratio, orderby_free_memory=True, exclude_gpus=exclude_gpus)

    session = hand_set_gpus(gpus=gpus, memory_ratio=ratio, auto_growth=auto_growth)
    return gpus, session


def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part * L:]
    return x[part * L:(part + 1) * L]


# keras自带的多GPU训练
def multi_gpu_train_model(model, num_gpus):
    def get_slice(data, i, parts):
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == num_gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)

    if num_gpus == 1:
        return model

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for gpu_id in range(num_gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('replica_%d' % gpu_id):
                inputs = []
                # Retrieve a slice of the input.
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]

                    slice_i = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': num_gpus, 'part': gpu_id})(x)
                    # slice_i = Lambda(get_slice, output_shape=input_shape, arguments={'i': i, 'parts': num_gpus})(x) #用这句报错
                    inputs.append(slice_i)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for name, outputs in zip(model.output_names, all_outputs):
            merged.append(concatenate(outputs,
                                      axis=0, name=name))
        return Model(model.inputs, merged)


# 要使用多GPU训练,必须保证generator每次均能返回GPU数量整数倍的batch size;
# 否则会报can't convert BatchDescriptor类似的错.

# gpus, session = set_gpus(num_gpus=2, free_ratio=0.3, exclude_gpus=[])
# print(gpus.split(",")[0], gpus.split(",")[1])
# print([]+gpus.split(","))
# set_gpus(num_gpus=1, free_ratio=0.3, exclude_gpus=["9"])
