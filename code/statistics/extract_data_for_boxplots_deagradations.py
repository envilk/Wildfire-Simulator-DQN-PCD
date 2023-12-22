# Import libraries
import numpy
import os
import json

data = []
labels = []
# Temporarily developed this way. Its getting all projects starting with "Wildfire-Simulator-DQN-PCD", from
# inside of one of the copies
for dirpath, dirnames, filenames in os.walk("../../../"):
    print(dirpath)
    if (dirpath.startswith("../../../Wildfire-Simulator-DQN-PCD") and dirpath.endswith("results/PCD/inference_results")):
        labels.append(dirpath)

labels.sort()
print(labels)

def pairwise(iterable):
    "s -> (s0, s1, s2), (s3, s4, s5), (s6, s7, s8), ..."
    a = iter(iterable)
    return zip(a, a, a)


num_UAV = 3

LOW_MR1 = []
MEDIUM_MR1 = []
HIGH_MR1 = []
LOW_MR2 = []
MEDIUM_MR2 = []
HIGH_MR2 = []

for num_dir, (dirname_LOW, dirname_MEDIUM, dirname_HIGH) in enumerate(pairwise([str(dir_) for dir_ in labels])):
    for UAV_idx in range(0, num_UAV):
        to_insert_LOW_MR1 = []
        to_insert_MEDIUM_MR1 = []
        to_insert_HIGH_MR1 = []
        to_insert_LOW_MR2 = []
        to_insert_MEDIUM_MR2 = []
        to_insert_HIGH_MR2 = []

        f_MR1 = str(UAV_idx+1) + 'UAV.txt'
        f_MR2 = 'COUNTER_' + str(UAV_idx + 1) + 'UAV.txt'

        print('--------')
        print(os.path.join(dirname_LOW, f_MR1))
        print(os.path.join(dirname_MEDIUM, f_MR1))
        print(os.path.join(dirname_HIGH, f_MR1))
        print(os.path.join(dirname_LOW, f_MR2))
        print(os.path.join(dirname_MEDIUM, f_MR2))
        print(os.path.join(dirname_HIGH, f_MR2))

        f_MR1_LOW_opened = os.path.join(dirname_LOW, f_MR1)
        f_MR1_MEDIUM_opened = os.path.join(dirname_MEDIUM, f_MR1)
        f_MR1_HIGH_opened = os.path.join(dirname_HIGH, f_MR1)
        f_MR2_LOW_opened = os.path.join(dirname_LOW, f_MR2)
        f_MR2_MEDIUM_opened = os.path.join(dirname_MEDIUM, f_MR2)
        f_MR2_HIGH_opened = os.path.join(dirname_HIGH, f_MR2)

        lines_MR1_LOW = open(f_MR1_LOW_opened).read().splitlines()
        lines_MR1_MEDIUM = open(f_MR1_MEDIUM_opened).read().splitlines()
        lines_MR1_HIGH = open(f_MR1_HIGH_opened).read().splitlines()
        lines_MR2_LOW = open(f_MR2_LOW_opened).read().splitlines()
        lines_MR2_MEDIUM = open(f_MR2_MEDIUM_opened).read().splitlines()
        lines_MR2_HIGH = open(f_MR2_HIGH_opened).read().splitlines()

        for UAV_inner_idx in range(0, UAV_idx+1):
            to_insert_LOW_MR1.append([json.loads(i)[UAV_inner_idx] for i in lines_MR1_LOW])
            to_insert_MEDIUM_MR1.append([json.loads(i)[UAV_inner_idx] for i in lines_MR1_MEDIUM])
            to_insert_HIGH_MR1.append([json.loads(i)[UAV_inner_idx] for i in lines_MR1_HIGH])
            to_insert_LOW_MR2.append([json.loads(i) for i in lines_MR2_LOW])
            to_insert_MEDIUM_MR2.append([json.loads(i) for i in lines_MR2_MEDIUM])
            to_insert_HIGH_MR2.append([json.loads(i) for i in lines_MR2_HIGH])

        to_insert_LOW_MR1_means = numpy.array(to_insert_LOW_MR1)
        to_insert_MEDIUM_MR1_means = numpy.array(to_insert_MEDIUM_MR1)
        to_insert_HIGH_MR1_means = numpy.array(to_insert_HIGH_MR1)
        to_insert_LOW_MR2_flatten = numpy.array(to_insert_LOW_MR2)
        to_insert_MEDIUM_MR2_flatten = numpy.array(to_insert_MEDIUM_MR2)
        to_insert_HIGH_MR2_flatten = numpy.array(to_insert_HIGH_MR2)

        to_insert_LOW_MR1_means = list(numpy.average(to_insert_LOW_MR1_means, axis=0))
        to_insert_MEDIUM_MR1_means = list(numpy.average(to_insert_MEDIUM_MR1_means, axis=0))
        to_insert_HIGH_MR1_means = list(numpy.average(to_insert_HIGH_MR1_means, axis=0))
        to_insert_LOW_MR2_flatten = to_insert_LOW_MR2_flatten.tolist()[0]
        to_insert_MEDIUM_MR2_flatten = to_insert_MEDIUM_MR2_flatten.tolist()[0]
        to_insert_HIGH_MR2_flatten = to_insert_HIGH_MR2_flatten.tolist()[0]

        LOW_MR1.append(to_insert_LOW_MR1_means)
        MEDIUM_MR1.append(to_insert_MEDIUM_MR1_means)
        HIGH_MR1.append(to_insert_HIGH_MR1_means)
        LOW_MR2.append(to_insert_LOW_MR2_flatten)
        MEDIUM_MR2.append(to_insert_MEDIUM_MR2_flatten)
        HIGH_MR2.append(to_insert_HIGH_MR2_flatten)

print(LOW_MR1)
print(MEDIUM_MR1)
print(HIGH_MR1)
print(LOW_MR2)
print(MEDIUM_MR2)
print(HIGH_MR2)