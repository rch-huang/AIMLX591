import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
import csv
import os
from dataset.tinyimagenet import TinyImagenet

from collections import Counter
import copy
import math
import random
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def count_digits_in_chunks(filename, cls=10,chunk_size=256):
    df = pd.read_csv(filename, header=None)
    
    data = df[0].astype(int).values
    
    num_chunks = len(data) // chunk_size
    if len(data) % chunk_size != 0:
        num_chunks += 1
    
    digit_counts = {i: [] for i in range(cls)}
    
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(data))
        chunk = data[start_index:end_index]
        
        counts = Counter(chunk)
        
        for digit in range(cls):
            digit_counts[digit].append(counts.get(digit, 0))
    
    return digit_counts, num_chunks

def plot_digit_statistics(filename,plot_dir, cls=10,chunk_size=256):
    digit_counts, num_chunks = count_digits_in_chunks(filename,cls, chunk_size)
    
    x = np.arange(num_chunks)
    
    plt.figure(figsize=(12, 8))
    for digit in range(cls):
        plt.plot(x, digit_counts[digit], label=f'class {digit}')
    
    plt.xlabel('sampling timeframe (1k)')
    plt.ylabel('occurence / timeframe')
    plt.title('')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir)
    #plt.show()





head_ratio = 0.001
head_weight = 0.10
steepness = 1
avoid_overlap = True

def long_tailed_prob(x,z, y, head_num, gamma=0.1,steepness=10):
    if not (x <= y <= z):
        raise ValueError("Ensure that x <= y <= z.")
    theta = head_num / (z-x)

    if not (0 <= theta <= 1) or not (0 <= gamma <= 1):
        raise ValueError("Ensure that 0 <= theta <= 1 and 0 <= gamma <= 1.")
    critical_point = x + (z - x) * theta
    
    if y < critical_point:
        prob_value = gamma + (1 - gamma) * (1 - (y - x) / (critical_point - x))
    else:
        tail_scale = 1 / (z - critical_point)
        prob_value = gamma * math.exp(-tail_scale * (y - critical_point)*steepness)
    
    return prob_value


def long_tailed_redistribution(sample_idx,opt):
    final_sample_idx = []
    final_sample_labels = []
    counter = {}
    total_counts = 0
    for s in range(len(sample_idx)):
         counter[s] = len(sample_idx[s])
         total_counts = total_counts + counter[s]
    if False:
        while True:
            for s in range(len(sample_idx)):
                if len(final_sample_idx) == total_counts:
                    return final_sample_idx,final_sample_labels
                final_sample_idx.append(sample_idx[s].pop())
                final_sample_labels.append(s)
    cls = sorted(counter.keys())
    bucks = {key: [0] for key in counter}
    indices_first_appearence = {key: -1 for key in counter}
    for i in range(total_counts):
        if len(final_sample_idx) == total_counts:
            break
        for j in range(len(cls)):
            cl = cls[j]
            cond = True
            if avoid_overlap:
                cond = len(sample_idx[cl]) >= counter[cl] * (1-head_ratio)
            else:
                cond = len(bucks[cl]) <= counter[cl] * head_ratio
            if  cond:
                index_for_first_appearence = indices_first_appearence[cl]
                if index_for_first_appearence == -1:
                    index_for_first_appearence = indices_first_appearence[cl] = i
                prob = long_tailed_prob(index_for_first_appearence,total_counts,i,head_ratio*counter[cl],head_weight,steepness)
                bucks[cl].append(prob)
                break
            else: 
                index_for_first_appearence = indices_first_appearence[cl]
                prob = long_tailed_prob(index_for_first_appearence,total_counts,i,head_ratio*counter[cl],head_weight,steepness)
                bucks[cl].append(prob)
                
        while True:
            weights = [bucks[i][-1] for i in bucks.keys()]
            total_weights = sum(weights)
            normalized = [w / total_weights for w in weights]
            selected = np.random.choice(list(range(len(cls))),p=normalized) 
            if len(sample_idx[selected])>0:
                for bb in range(1):
                    final_sample_idx.append(sample_idx[selected].pop())
                    final_sample_labels.append(selected)
                break
        
        #break
    print(final_sample_labels)
    #final_sample_idx.clear()
    #final_sample_labels.clear()
    # for i in range(40960):
    #             selected = np.random.choice(list(range(len(sample_idx[2])))) 
    #             final_sample_idx.append(sample_idx[2][selected])
    #             final_sample_labels.append(2)
    #             print(i)
    #print([str(label)+"," for label in final_sample_labels])
    with open('./final_sample_labels.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in final_sample_labels:
            writer.writerow([item])
    plot_digit_statistics('./final_sample_labels.csv',
                          os.path.join(opt.save_folder, 'sampling_dist_'+str(head_ratio)+"_"+str(head_weight)+"_"+str(steepness)+""+str(avoid_overlap)+".png"),
                          len(sample_idx))
    return final_sample_idx,final_sample_labels  
        
def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.'
    Code copied from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


class ThreeCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, size):
        self.transform = transform
        self.notaug_transform = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.notaug_transform(x)]


class SeqSampler(Sampler):
    def __init__(self, dataset, blend_ratio, n_concurrent_classes,
                 train_samples_per_cls,opt):
        """data_source is a Subset"""
        self.num_samples = len(dataset)
        self.blend_ratio = blend_ratio
        self.n_concurrent_classes = n_concurrent_classes
        self.train_samples_per_cls = train_samples_per_cls
        if opt.longtailed:
            self.longtailed = True
        else:
            self.longtailed = False
        copy_opt = copy.deepcopy(opt)
        copy_opt.trial = 0
        self.opt = copy_opt         
        # Configure the correct train_subset and val_subset
        if torch.is_tensor(dataset.targets):
            self.labels = dataset.targets.detach().cpu().numpy()
        else:  # targets in cifar10 and cifar100 is a list
            self.labels = np.array(dataset.targets)
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)

    def __iter__(self):
        """Sequential sampler"""
        # Configure concurrent classes
        cmin = []
        cmax = []
        for i in range(int(self.n_classes / self.n_concurrent_classes)):
            for _ in range(self.n_concurrent_classes):
                cmin.append(i * self.n_concurrent_classes)
                cmax.append((i + 1) * self.n_concurrent_classes)
        print('cmin', cmin)
        print('cmax', cmax)

        filter_fn = lambda y: np.logical_and(
            np.greater_equal(y, cmin[c]), np.less(y, cmax[c]))
        
        filter_fn2 = lambda y: np.logical_and(
            np.greater_equal(y, c), True)
        # Configure sequential class-incremental input
        sample_idx = []
        for c in self.classes:
            filtered_train_ind = None
            if c >= len(cmax):
                filtered_train_ind = filter_fn2(self.labels)
            else:
                filtered_train_ind = filter_fn(self.labels)
            filtered_ind = np.arange(self.labels.shape[0])[filtered_train_ind]
            np.random.shuffle(filtered_ind)

            cls_idx = self.classes.index(c)
            if len(self.train_samples_per_cls) == 1:  # The same length for all classes
                sample_num = self.train_samples_per_cls[0]
            else:  # Imbalanced class
                assert len(self.train_samples_per_cls) == len(self.classes), \
                    'Length of classes {} does not match length of train ' \
                    'samples per class {}'.format(len(self.classes),
                                                  len(self.train_samples_per_cls))
                sample_num = self.train_samples_per_cls[cls_idx]

            sample_idx.append(filtered_ind.tolist()[:sample_num])
            if c >= len(cmin):
                print('Class [{}, {}): {} samples'.format(cls_idx, cls_idx,
                                                      sample_num))
            else:
                print('Class [{}, {}): {} samples'.format(cmin[cls_idx], cmax[cls_idx],
                                                      sample_num))

        # Configure blending class
        if self.blend_ratio > 0.0:
            for c in range(len(self.classes)):
                # Blend examples from the previous class if not the first
                if c > 0:
                    blendable_sample_num = \
                        int(min(len(sample_idx[c]), len(sample_idx[c-1])) * self.blend_ratio / 2)
                    # Generate a gradual blend probability
                    blend_prob = np.arange(0.5, 0.05, -0.45 / blendable_sample_num)
                    assert blend_prob.size == blendable_sample_num, \
                        'unmatched sample and probability count'

                    # Exchange with the samples from the end of the previous
                    # class if satisfying the probability, which decays
                    # gradually
                    for ind in range(blendable_sample_num):
                        if random.random() < blend_prob[ind]:
                            tmp = sample_idx[c-1][-ind-1]
                            sample_idx[c-1][-ind-1] = sample_idx[c][ind]
                            sample_idx[c][ind] = tmp
        for cls in sample_idx:
                print(len(cls))
        if self.longtailed:
            final_idx,_ = self.long_tailed_redistribution2(sample_idx,self.opt)
            return iter(final_idx)
        else:   
            final_idx = []
            for sample in sample_idx:
                final_idx += sample
            return iter(final_idx)
    
    def long_tailed_redistribution2(self,sample_idx,opt):
        final_sample_idx = []
        num_of_classes = len(sample_idx)
        remaining_by_class = [len(sample_idx[i]) for i in range(num_of_classes)]
        total_size_by_class = [len(sample_idx[i]) for i in range(num_of_classes)]
        batch_size = 1024
        while len(final_sample_idx) < sum(total_size_by_class):
            batch_idx = []
            for i in range(num_of_classes):
                if remaining_by_class[i] >= 0.3 * total_size_by_class[i]:
                    for j in range(int(batch_size*0.8)):
                        #do with the probability of 0.8
                        if random.random() < 0.75:
                            batch_idx.append(sample_idx[i].pop())
                            remaining_by_class[i] -= 1
                            if remaining_by_class[i] == 0:
                                break
                    if True: 
                        total_remaining_before_i = sum(remaining_by_class[:i])
                        for k in range(i):
                            if True:
                                t = remaining_by_class[k] 
                                for l in range(t):
                                    if random.random() < 1.2*remaining_by_class[k]/total_size_by_class[k]:
                                        batch_idx.append(sample_idx[k].pop())
                                        remaining_by_class[k] -= 1
                    break
            final_sample_idx += batch_idx
            if len(batch_idx) == 0:
                break
        return final_sample_idx,None

        
    def __len__(self):
        if len(self.train_samples_per_cls) == 1:
            return self.n_classes * self.train_samples_per_cls[0]
        else:
            return sum(self.train_samples_per_cls)

def redistribute_samples(sample_idx):
    n = len(sample_idx)

    for i in range(n):
        current_samples = sample_idx[i]

        num_to_move = 300

        if num_to_move == 0:
            continue

        samples_to_move = random.sample(current_samples, num_to_move)

        for offset in [3, 5, 7]:
            if i + offset < n:
                sample_idx[i + offset].extend(samples_to_move)
        
        sample_idx[i] = [x for x in current_samples if x not in samples_to_move]

    return sample_idx


def set_loader(opt):
    # set seed for reproducing
    opt_trial = 0
    random.seed(opt_trial)
    np.random.seed(opt_trial)
    torch.manual_seed(opt_trial)

    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tinyimagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    elif opt.dataset == 'path':
        mean = opt.mean
        std = opt.std
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_runtime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=ThreeCropTransform(train_transform, opt.size),
                                         download=True,
                                         train=True)
        knn_train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                             train=True,
                                             transform=val_transform)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_runtime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=ThreeCropTransform(train_transform, opt.size),
                                          download=True,
                                          train=True)
        knn_train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                              train=True,
                                              transform=val_transform)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)

        # Convert sparse labels to coarse labels
        train_dataset.targets = sparse2coarse(train_dataset.targets)
        knn_train_dataset.targets = sparse2coarse(knn_train_dataset.targets)
        val_dataset.targets = sparse2coarse(val_dataset.targets)

    elif opt.dataset == 'tinyimagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size,
                                         scale=(0.08, 1.0),
                                         ratio=(3.0 / 4.0, 4.0 / 3.0),
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_runtime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=opt.size,
                                         scale=(0.08, 1.0),
                                         ratio=(3.0 / 4.0, 4.0 / 3.0),
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = TinyImagenet(root=opt.data_folder + 'TINYIMG',
                                     transform=ThreeCropTransform(train_transform, opt.size),
                                     train=True,
                                     download=True)
        knn_train_dataset = TinyImagenet(root=opt.data_folder + 'TINYIMG',
                                         train=True,
                                         transform=val_transform)
        val_dataset = TinyImagenet(root=opt.data_folder + 'TINYIMG',
                                   train=False,
                                   transform=val_transform)

    elif opt.dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.Resize(size=opt.size),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_runtime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=opt.size),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.MNIST(root=opt.data_folder,
                                       transform=ThreeCropTransform(train_transform, opt.size),
                                       download=True,
                                       train=True)
        knn_train_dataset = datasets.MNIST(root=opt.data_folder,
                                           train=True,
                                           transform=val_transform)
        val_dataset = datasets.MNIST(root=opt.data_folder,
                                     train=False,
                                     transform=val_transform)

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                             transform=val_transform)
        knn_train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                 transform=val_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder,
                                           transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    # Configure the a smaller subset as validation dataset
    if torch.is_tensor(train_dataset.targets):
        labels = train_dataset.targets.detach().cpu().numpy()
    else:  # targets in cifar10 and cifar100 is a list
        labels = np.array(train_dataset.targets)
    num_labels = len(list(set(labels)))

    # Create training loader
    if opt.training_data_type == 'iid':
        train_subset_len = num_labels * opt.train_samples_per_cls[0]
        train_subset, _ = torch.utils.data.random_split(dataset=train_dataset,
                                                        lengths=[train_subset_len,
                                                                 len(train_dataset) - train_subset_len])
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)
    else:  # sequential
        train_sampler = SeqSampler(train_dataset, opt.blend_ratio,
                                   opt.n_concurrent_classes,
                                   opt.train_samples_per_cls,opt)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    # Create validation loader
    val_subset_len = num_labels * opt.test_samples_per_cls
    val_subset, _ = torch.utils.data.random_split(dataset=val_dataset,
                                                  lengths=[val_subset_len, len(val_dataset) - val_subset_len])
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=opt.val_batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    # Create kNN loader
    if opt.knn_samples > 0:
        knn_subset, _ = torch.utils.data.random_split(dataset=knn_train_dataset,
                                                      lengths=[opt.knn_samples, len(knn_train_dataset) - opt.knn_samples])
        knn_train_loader = torch.utils.data.DataLoader(knn_subset,
                                                      batch_size=opt.val_batch_size,
                                                      shuffle=False,
                                                      num_workers=0, pin_memory=True)
    else:
        knn_train_loader = None

    print('Training samples: ', len(train_loader) * opt.batch_size)
    print('Testing samples: ', len(val_loader) * opt.val_batch_size)
    print('kNN training samples: ', len(knn_train_loader) * opt.val_batch_size)

    return train_loader, val_loader, knn_train_loader, train_transform_runtime
