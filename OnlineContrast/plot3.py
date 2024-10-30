import matplotlib.pyplot as plt
import itertools
import json
import sys

import numpy as np
import random

cls_samples_size = 4096
cls_samples_array = [4096,4096*5,4096,4096,4096*4,4096*3,4096*2]
#cls_samples_array = [cls_samples_size for i in range(10)]
number_of_classes = len(cls_samples_array)

batch_size = 1024
samples_seq = []
head_ratio = 0.8


def fun1():
    global number_of_classes
    sampled_array = np.array(samples_seq)
    for cls in range(number_of_classes):
        if cls_samples_array[cls]*head_ratio*0.8 > np.sum(sampled_array==cls):
            return cls
    return None


def fun2(current_cls):
    dist_prob = []
    dist_cls = []
    sampled_array = np.array(samples_seq)
    for i in range(number_of_classes):
        if i != current_cls:
            count = np.sum(sampled_array==i)
            if count < cls_samples_array[i] * 1.05:
                dist_prob.append(max(1,count))
                dist_cls.append(i)
    dist_prob = np.array(dist_prob)
    dist_prob = [i / (np.sum(dist_prob)) for i in dist_prob]
    dist_prob = [i / np.sum(np.power(dist_prob,0.5)) for i in np.power(dist_prob,0.5)]
    # dist_prob = [i / np.sum(np.exp(np.array(dist_prob))) for i in np.exp(np.array(dist_prob))]
    return dist_cls,dist_prob
            




def fun(samples):
    range_a = 0
    range_b = 0
    pre_slots = [0 for i in range(number_of_classes)]
    cur_slots = [0 for i in range(number_of_classes)]
    while True:
        pre_slots = cur_slots
        cur_slots = [0 for i in range(number_of_classes)]
        sampled_array = np.array(samples_seq)
        range_a = len(samples_seq)
        cls_filled_its_head = fun1()
        if cls_filled_its_head != None:
            dist_cls,dist_prob = fun2(cls_filled_its_head)
            to_fill = cls_samples_array[cls_filled_its_head]*head_ratio - np.sum(sampled_array==cls_filled_its_head)
            filled = 0
            to_fill = min(batch_size,to_fill)
            while to_fill>0:
                to_fill = to_fill -1
                r = random.randint(0,batch_size)
                if r <= batch_size*0.70:
                    filled = filled + 1
                    samples_seq.append(cls_filled_its_head)
            remained = batch_size - filled
            # remained = np.random.choice(dist_cls,size = int(remained),p=dist_prob)
            while remained>0:
                if len(dist_cls)==0:
                    break
                j = np.random.choice(dist_cls,p=dist_prob)
                if pre_slots[j]!=0 and (cur_slots[j]>pre_slots[j]*1.05):
                    samples_seq.append(cls_filled_its_head)
                    remained = remained -1
                    j_index = dist_cls.index(j)
                    dist_cls.remove(j)
                    dist_prob.remove(dist_prob[j_index])
                    dist_prob = [ii/np.sum(np.array(dist_prob)) for ii in  dist_prob ]
                    continue
                if np.sum(np.array(samples_seq)==j) < cls_samples_array[i]*1.05:
                        cur_slots[j] = cur_slots[j] + 1
                        samples_seq.append(j)
                        remained = remained -1
                else:
                    j_index = dist_cls.index(j)
                    dist_cls.remove(j)
                    dist_prob.remove(dist_prob[j_index])
                    dist_prob = [ii/np.sum(np.array(dist_prob)) for ii in  dist_prob ]
            range_b = len(samples_seq)
            sampled_range = samples_seq[range_a:range_b]
            random.shuffle(sampled_range)
            samples_seq[range_a:range_b] = sampled_range
        else:
            return samples_seq


for i in range(1):
    samples_seq = []
    samples_seq = fun([])
    unique, counts = np.unique(samples_seq, return_counts=True)
    print(counts)


 
import matplotlib.pyplot as plt

# Example array
arr = np.array(samples_seq)

# Interval size
n = 512

# Initialize a list to store counts for each interval
interval_counts = []

# Loop over the array in intervals of size n
for i in range(0, len(arr), n):
    # Take the interval (chunk)
    chunk = arr[i:i+n]
    
    # Count the occurrences of each digit 0-9 in the chunk
    counts = [np.sum(chunk == digit) for digit in range(10)]
    
    # Store the counts
    interval_counts.append(counts)

# Convert interval_counts to a numpy array for easier plotting
interval_counts = np.array(interval_counts)

# Create x-axis points (one point per interval)
x = np.arange(len(interval_counts))

# Plot the counts for each digit
for digit in range(number_of_classes):
    plt.plot(x, interval_counts[:, digit], label=f'Digit {digit}')

# Add labels and legend
plt.xlabel('Interval')
plt.ylabel('Count')
plt.legend()
plt.title(f'Counts of digits 0-9 in intervals of size {n}')
plt.show()
