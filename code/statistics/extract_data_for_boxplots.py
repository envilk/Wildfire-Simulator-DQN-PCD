# Import libraries
import matplotlib.pyplot as plt
import pandas
import numpy
import plotly.express as px
import os
import json

data_DQN_MR1 = []
data_PCD_MR1 = []
data_DQN_MR2 = []
data_PCD_MR2 = []
labels = []
for dirpath, dirnames, filenames in os.walk("."):
    if dirpath.startswith("./pruebas_") and dirpath.endswith("0"):
        print(dirpath)
        labels.append(dirpath)

labels.sort()
print(labels)

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

num_UAV = 3
num_lines = 0
num_dir_outter = 0
for num_dir, (dirname_entrenamiento, dirname_predictor) in enumerate(pairwise([str(dir_) for dir_ in labels])):
    for UAV_idx in range(0, num_UAV):
        to_insert_e_MR1 = []
        to_insert_e_MR2 = []
        to_insert_p_MR1 = []
        to_insert_p_MR2 = []

        f_MR1 = '0_' + str(UAV_idx+1) + 'UAV.txt'
        f_MR2 = 'COUNTER_' + str(UAV_idx + 1) + 'UAV.txt'

        print('--------')
        print(os.path.join(dirname_entrenamiento, f_MR1))
        print(os.path.join(dirname_predictor, f_MR2))
        print(os.path.join(dirname_entrenamiento, f_MR1))
        print(os.path.join(dirname_predictor, f_MR2))

        f_e_opened_MR1 = os.path.join(dirname_entrenamiento, f_MR1)
        f_e_opened_MR2 = os.path.join(dirname_entrenamiento, f_MR2)
        f_p_opened_MR1 = os.path.join(dirname_predictor, f_MR1)
        f_p_opened_MR2 = os.path.join(dirname_predictor, f_MR2)

        lines_e_MR1 = open(f_e_opened_MR1).read().splitlines()
        lines_e_MR2 = open(f_e_opened_MR2).read().splitlines()
        lines_p_MR1 = open(f_p_opened_MR1).read().splitlines()
        lines_p_MR2 = open(f_p_opened_MR2).read().splitlines()

        for UAV_inner_idx in range(0, UAV_idx+1):
            to_insert_e_MR1.append([json.loads(i)[UAV_inner_idx] for i in lines_e_MR1])
            to_insert_e_MR2.append([json.loads(i) for i in lines_e_MR2])
            to_insert_p_MR1.append([json.loads(i)[UAV_inner_idx] for i in lines_p_MR1])
            to_insert_p_MR2.append([json.loads(i) for i in lines_p_MR2])
        data_MR1 = []
        to_insert_e_MR1_means = numpy.array(to_insert_e_MR1)
        to_insert_e_MR2_flatten = numpy.array(to_insert_e_MR2)
        to_insert_p_MR1_means = numpy.array(to_insert_p_MR1)
        to_insert_p_MR2_flatten = numpy.array(to_insert_p_MR2)

        to_insert_e_MR1_means = list(numpy.average(to_insert_e_MR1_means, axis=0))
        to_insert_e_MR2_flatten = to_insert_e_MR2_flatten.tolist()[0]
        to_insert_p_MR1_means = list(numpy.average(to_insert_p_MR1_means, axis=0))
        to_insert_p_MR2_flatten = to_insert_p_MR2_flatten.tolist()[0]

        data_DQN_MR1.append(to_insert_e_MR1_means)
        data_PCD_MR1.append(to_insert_p_MR1_means)
        data_DQN_MR2.append(to_insert_e_MR2_flatten)
        data_PCD_MR2.append(to_insert_p_MR2_flatten)

print(data_DQN_MR1)
print(data_PCD_MR1)
print(data_DQN_MR2)
print(data_PCD_MR2)