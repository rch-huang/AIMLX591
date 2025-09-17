from __future__ import print_function
import torch
import os
import copy
import sys
import argparse
import time
import numpy as np
import random
from tqdm import tqdm
import json
import tensorboard_logger as tb_logger 
 
from util import AverageMeter

from memory import Memory
from losses.supcon import similarity_mask_new, similarity_mask_old  # for scale
from losses import get_loss
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from eval_utils import knn_eval, knn_task_eval, cluster_eval, plot_mem_select, \
    plot_mem, save_model
from data_utils import set_loader
from set_utils import load_student_backbone, set_optimizer, set_constant_learning_rate

VAL_CNT = 10       # Number of validations to allow during training
SAVE_MODEL = False # Whether to save model for offline linear evaluation
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 20,
    'tinyimagenet': 10,  # temp setting
    'mnist': 10
}

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    #parser.add_argument('--plot_freq', type=int, default=5000,
    #                    help='plot frequency of steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='batch_size in validation')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--steps_per_batch_stream', type=int, default=20,
                        help='number of steps for per batch of streaming data')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')

    # optimization
    parser.add_argument('--learning_rate_stream', type=float, default=0.1,
                        help='learning rate for streaming new data')
    parser.add_argument('--learning_rate_pred', type=float, default=0.01,
                        help='learning rate for predictor g')
    # parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
    #                    help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1,
    #                    help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--distill_power', type=float, default=0.1)


    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'tinyimagenet', 'path'],
                        help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--training_data_type', type=str, default='iid',
                        choices=['iid', 'class_iid', 'instance', 'class_instance'],
                        help='iid or sequential datastream')
    parser.add_argument('--blend_ratio', type=float, default=.0,
                        help="the ratio blend classes at the boundary")
    parser.add_argument('--n_concurrent_classes', type=int, default=1,
                        help="the number of concurrent classes showing at the same time")
    parser.add_argument('--train_samples_per_cls', type=int, nargs='+',
                        help="the number of training samples per class")
    parser.add_argument('--test_samples_per_cls', type=int, default=1000,
                        help="the number of testing samples per class")
    parser.add_argument('--knn_samples', type=int, default=0,
                        help='number of labeled samples loaded for kNN training')
    parser.add_argument('--kneighbor', type=int, default=50,
                        help="the number of neighbors in knn")


    # method
    parser.add_argument('--criterion', type=str, default='supcon',
                        choices=['supcon', 'simclr', 'scale'],
                        help='major criterion')
    parser.add_argument('--mask_memory', type=int, default=1,
                        help="whether mask memory samples in contrastive loss")
    parser.add_argument('--lifelong_id', type=int, default=0,
                        help='id for lifelong learning method') #0 none, 2 scale, 1 co2l    
                                             
    parser.add_argument('--distill_enabled', type=str, default='none',
                        choices=['none', 'scale', 'co2l'],
                        help='choose lifelong learning method')
    parser.add_argument('--simil', type=str, default='tSNE',
                        choices=['tSNE', 'kNN'],
                        help='choose similarity metric')
    parser.add_argument('--longtailed', type=int, default=0,
                        help="whether use labels during memory update")
    # temperature
    parser.add_argument('--temp_cont', type=float, default=0.1,
                        help='temperature for contrastive loss function')
    parser.add_argument('--temp_tSNE', type=float, default=0.1,
                        help='temperature for tSNE similarity')
    parser.add_argument('--thres_ratio', type=float, default=0.3,
                        help='threshold ratio between mean and max')
    parser.add_argument('--current_temp', type=float, default=0.1,
                        help='temperature for loss function')
    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')
    
    # other setting
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to the pretrained backbone to load')
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')
    parser.add_argument('--testid', type=int, default=0,
                        help='test Index for AIMLX 591 report')
    # Memory
    parser.add_argument('--mem_max_classes', type=int, default=1,
                        help='max number of classes in memory')
    parser.add_argument('--mem_size', type=int, default=128,
                        help='number of samples per class in memory')
    parser.add_argument('--mem_samples', type=int, default=0,
                        help='number of samples to add during each training step')
    parser.add_argument('--mem_update_type', type=str, default='reservoir',
                        choices=['rdn', 'mo_rdn', 'reservoir', 'simil'],
                        help='memory update policy')
    parser.add_argument('--normalize_embeddings', type=int, default=0,
                        help="whether use normalized embeddings")
    parser.add_argument('--mem_w_labels', type=int, default=0,
                        help="whether use labels during memory update")
    parser.add_argument('--mem_cluster_type', type=str, default='none',
                        choices=['none', 'kmeans', 'spectral', 'max_coverage',
                                 'psa', 'maximin', 'energy'],
                        help="which clustering method to use during unsupervised update")
    parser.add_argument('--mem_max_new_ratio', type=float, default=0.1,
                        help='max ratio of new samples at each memory update')

    # Evaluation
    parser.add_argument('--k_scale', type=float, default=1.0,
                        help='to scale the number of classes during evaluation')
    parser.add_argument('--plot', default=False, action="store_true",
                        help="whether to plot during evaluation")
    parser.add_argument('--clofai_prefix', type=str, default='clofai',)
    opt = parser.parse_args()
    
     
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

 
    if opt.data_folder is None:
        #raise ValueError('need to specify data path for dataset {}'.format(opt.dataset))
        opt.data_folder = '/Scratch/repository/rh539/'#'../datasets/'
    opt.model_path = './save/{}_models/'.format(opt.dataset)
    opt.tb_path = './save/{}_tensorboard/'.format(opt.dataset)

     
    if len(opt.train_samples_per_cls) == 1:
        im_ind = opt.train_samples_per_cls[0]
    else:
        im_ind = 'im'
    from datetime import datetime
    opt.model_name = 'Model_{}_{}_{}_{}_{}_{}_{}_{}_s{}_{}_{}_lrs_{}_bsz_{}_mem_{}_{}_{}_{}_{}_{}_temp_{}_' \
                     'simil_{}_{}_{}_distill_{}_{}_{}_steps_{}_epoch_{}_trial_{}'.format(
        datetime.now().strftime("%m%d%H%M%S") ,               
        opt.distill_enabled, opt.criterion, opt.dataset, opt.model,
        opt.training_data_type, opt.blend_ratio, opt.n_concurrent_classes,
        im_ind, opt.test_samples_per_cls, opt.knn_samples,
        opt.learning_rate_stream, opt.batch_size, opt.mem_samples,
        opt.mem_size, opt.mem_max_classes, opt.mem_update_type, opt.mem_cluster_type, int(opt.mem_w_labels),
        opt.temp_cont, opt.simil, opt.temp_tSNE, opt.thres_ratio,
        opt.distill_power, opt.current_temp, opt.past_temp,
        opt.steps_per_batch_stream, opt.epochs, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)


    opt.logfilename = 'Date_'+str(datetime.now().strftime('%Y%m%d_%H_%M_%S'))+'#Criterion_'+str(opt.criterion)+'#LifelongMethod_'+str(opt.distill_enabled)+'#Steps_'+str(opt.steps_per_batch_stream)+"#BatchSize_"+str(opt.batch_size)+'#Epochs_'+str(opt.epochs)+'#Mem_'+str(opt.mem_samples)
    return opt


def train_step(images, labels, models, criterions, optimizer,
               meters, opt, mem, train_transform):
    """One gradient descent step"""
    model, past_model = models
    criterion, criterion_reg = criterions
    losses_stream, losses_contrast, losses_distill = meters

    # load memory samples
    if opt.mem_samples>0:
        mem_images, mem_labels,mem_true_labels = mem.get_mem_samples()

    # get augmented streaming and augmented samples and concatenate them
    if opt.mem_samples > 0 and mem_images is not None:
        if opt.mem_size == -1 :#Unlimited Memory Size   #40960 and opt.mem_w_labels and opt.criterion == 'supcon':
            # mem_true_labels_numpy = mem_true_labels.detach().numpy()
            # lb_set = set(mem_true_labels_numpy)
            # cls_count = [np.sum(mem_true_labels_numpy == i) for i in lb_set]
            # sample_cnt = min(cls_count)
            # sample_cnt = min(102, sample_cnt)
            # select_ind = []
            # #select_ind is randomly choice from each class to make sure each class has the same number of samples, aka sample_cnt
            # for i in lb_set:
            #     select_ind += np.random.choice(np.where(mem_true_labels_numpy == i)[0], sample_cnt, replace=False).tolist()
            # select_ind = np.array(select_ind)
            raise NotImplementedError("Unlimited Memory Size not implemented")
        else:
            sample_cnt = min(opt.mem_samples, mem_images.shape[0])
            select_ind = np.random.choice(mem_images.shape[0], sample_cnt, replace=False)
        mem_true_labels = mem_true_labels[select_ind]
      
        # Augment memory samples
        aug_mem_images_0 = torch.stack([train_transform(ee.cpu() * 255)
                                        for ee in mem_images[select_ind]])
        aug_mem_images_1 = torch.stack([train_transform(ee.cpu() * 255)
                                        for ee in mem_images[select_ind]])
        # restore 0-1 to 0-255

        feed_images_0 = torch.cat([images[0], aug_mem_images_0], dim=0)
        feed_images_1 = torch.cat([images[1], aug_mem_images_1], dim=0)
        feed_labels = torch.cat([labels, mem_labels[select_ind]], dim=0)
    else:
        feed_images_0 = images[0]
        feed_images_1 = images[1]
        feed_labels = labels

    if torch.cuda.is_available():
        feed_images_0 = feed_images_0.cuda(non_blocking=True)
        feed_images_1 = feed_images_1.cuda(non_blocking=True)
        feed_labels = feed_labels.cuda(non_blocking=True)

    
    bsz = feed_images_0.shape[0]
    
    model.train()

    loss_distill = .0
    if opt.distill_enabled == 'none':

        if opt.criterion == 'supcon':
            loss_contrast = criterion(model, model, feed_images_0, feed_images_1,
                             labels=feed_labels)

        elif opt.criterion in ['simclr', 'cka', 'barlowtwins', 'byol', 'vicreg', 'simsiam']:
            loss_contrast = criterion(model, model, feed_images_0, feed_images_1)

    elif opt.distill_enabled == 'scale':
        if opt.lifelong_id == 3 or opt.lifelong_id == 4:
            f0_logits, loss_distill = criterion_reg(model, past_model,
                                                feed_images_0)
            losses_distill.update(loss_distill.item(), bsz)
 
        #contrast_mask = similarity_mask_new(opt.batch_size, f0_logits, opt, pos_pairs)
        feed_images_all = torch.cat([feed_images_0, feed_images_1], dim=0)
        features_all = model(feed_images_all)
        contrast_mask = similarity_mask_old(features_all, bsz, opt)
        loss_contrast = criterion(model, model, feed_images_0, feed_images_1,
                                  mask=contrast_mask)

    elif opt.distill_enabled == 'co2l':

        f0_logits, loss_distill = criterion_reg(model, past_model,
                                                feed_images_0)
        losses_distill.update(loss_distill.item(), bsz)
        if opt.criterion == 'supcon':
            loss_contrast = criterion(model, model, feed_images_0, feed_images_1,
                             labels=feed_labels)
        elif opt.criterion == 'simclr':
            loss_contrast = criterion(model, model, feed_images_0, feed_images_1)
        else:
            raise ValueError('contrastive method not supported: {}'.format(opt.criterion))
     

    else:
        raise ValueError('contrastive method not supported: {}'.format(opt.distill_enabled))


    losses_contrast.update(loss_contrast.item(), bsz)

    if train_step.distill_power <= 0.0 and loss_distill > 0.0:
        train_step.distill_power = losses_contrast.avg * opt.distill_power / losses_distill.avg

    loss = loss_contrast + train_step.distill_power * loss_distill
    
    losses_stream.update(loss.item(), bsz)

    # SGD
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if opt.criterion == 'byol':
        criterion.update_moving_average()
    if opt.mem_samples>0:
        return mem_true_labels
    else:
        return

train_step.distill_power = 0.0

def convert_types(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
def train(task_loaders, test_loader, knntrain_loader,
          train_transform, model, criterions, optimizer,
          opt, mem, logger,epoch=1):
    
    
    
    losses_stream = AverageMeter()
    losses_contrast = AverageMeter()
    losses_distill = AverageMeter()
    pos_pairs_stream = AverageMeter()
    meters = [losses_stream, losses_contrast, losses_distill]

     
     
  
    seen_classes = [] 
     
    step_since_beginning = 0
    opt.epochs = 200
    for train_loader in task_loaders:
        for epoch in range(opt.epochs):
            print('Epoch: [{}/{}]'.format(epoch, opt.epochs) + ' Step since beginning: {}'.format(step_since_beginning))
            for idx, (images, labels) in enumerate(train_loader):
                # len(images) == 3, two augmented images and one original image


                labels_set = set(labels.tolist())
                for label in labels_set:
                    if label not in seen_classes:
                        seen_classes.append(label)
                if epoch == 0:
                    opt.stats["times_cls"][step_since_beginning] = {}
                    for label in labels:
                        _label = label.item()
                        if _label not in opt.stats["times_cls"][step_since_beginning]:
                            opt.stats["times_cls"][step_since_beginning][_label] = 0
                        opt.stats["times_cls"][step_since_beginning][_label] += 1 
                    print(str(step_since_beginning) +" cls "+str(opt.stats["times_cls"][step_since_beginning])) 
                     
                 
                # compute loss
                for step in range(opt.steps_per_batch_stream):
                    
                    
                     
                    past_model = copy.deepcopy(model)
                    past_model.eval()
                    models = [model, model]
                    
                     
                    mem_true_labels = train_step(images, labels, models, criterions,
                            optimizer, meters, opt, mem, train_transform)
                    if mem_true_labels is not None and step == 0 and epoch == 0:
                        opt.stats["times_mem_cls"][step_since_beginning] = {}
                        for mem_true_label in mem_true_labels:
                            _mem_true_label = mem_true_label.item()
                            if _mem_true_label not in opt.stats["times_mem_cls"][step_since_beginning]:
                                opt.stats["times_mem_cls"][step_since_beginning][_mem_true_label] = 0
                            opt.stats["times_mem_cls"][step_since_beginning][_mem_true_label] += 1 
                        print(str(step_since_beginning) +" mem "+str(opt.stats["times_mem_cls"][step_since_beginning])) 

                         
                    if step == opt.steps_per_batch_stream-1 and epoch == opt.epochs-1:
                        validate(test_loader, knntrain_loader, model, optimizer,
                            opt, seen_classes)
                    
                    logger.log_value('learning_rate',
                                    optimizer.param_groups[0]['lr'],
                                    step_since_beginning)
                    logger.log_value('loss',
                                    losses_stream.avg,
                                    step_since_beginning)
                    logger.log_value('loss_contrast',
                                    losses_contrast.avg,
                                    step_since_beginning)
                    logger.log_value('loss_distill',
                                    losses_distill.avg * train_step.distill_power,
                                    step_since_beginning)
                    logger.log_value('pos_pair_stream', pos_pairs_stream.avg,
                                    step_since_beginning)
                    step_since_beginning += 1

                    # print info
                    if step_since_beginning % 10 == 0 and False:
                        print('Train stream: [{step_idx}/{0}/{1}]\t'
                            'loss {2}\t'
                            'loss_contrast {3}\t'
                            'loss_distill {4}\t'
                            'pos pairs {pos_pair.val:.3f} ({pos_pair.avg:.3f})'.format(
                            idx + 1, len(train_loader),
                            losses_stream.avg,
                            losses_contrast.avg,
                            losses_distill.avg * train_step.distill_power,
                            step_idx=step_since_beginning,
                            pos_pair=pos_pairs_stream))
                        sys.stdout.flush()

                if epoch != 0:
                    # Skip memory update for subsequent epochs
                    continue

                if opt.mem_w_labels:
                    mem.update_w_labels(images[2], labels)
                   
                else:
                    all_embeddings, all_true_labels, select_indexes = \
                        mem.update_wo_labels(images[2], labels, model)
                    

                
            
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    norms[norms == 0] = 1
    
    normalized_embeddings = embeddings / norms
    
    return normalized_embeddings
def validate(test_loader, knn_train_loader, model, optimizer,
             opt,seen_classes=[]):
    model.eval()
    test_labels, knn_labels = [], []
    test_embeddings,   knn_embeddings = None, None 

    
    for _, (images, labels) in enumerate(tqdm(knn_train_loader, desc="knn training")):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        embeddings = model(images).detach().cpu().numpy()
        if opt.normalize_embeddings == 1:
            embeddings = normalize_embeddings(embeddings)
        if knn_embeddings is None:
            knn_embeddings = embeddings
        else:
            knn_embeddings = np.concatenate((knn_embeddings, embeddings), axis=0)
        knn_labels += labels.detach().tolist()
    knn_labels = np.array(knn_labels).astype(int)

    
    for idx, (images, labels) in enumerate(tqdm(test_loader, desc='test')):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        embeddings = model(images).detach().cpu().numpy()
        if test_embeddings is None:
            test_embeddings = embeddings
        else:
            test_embeddings = np.concatenate((test_embeddings, embeddings), axis=0)
        test_labels += labels.detach().tolist()
    test_labels = np.array(test_labels).astype(int)
    
    knn_eval(test_embeddings, test_labels, knn_embeddings, knn_labels,
             opt, seen_classes )
     

def main():
    opt = parse_option()

    print("============================================")
    print(opt)
    print("============================================")

    # set seed for reproducing

    random.seed(opt.trial)
    np.random.seed(opt.trial)
    torch.manual_seed(opt.trial)
    
    # build model
    model = load_student_backbone(opt.model,
                                  opt.dataset,
                                  opt.ckpt)
     
    task_loaders, test_loader, knntrain_loader, train_transform = set_loader(opt)

    
    
    criterion, criterion_reg = get_loss(opt)
    criterions = [criterion, criterion_reg]

    
    optimizer = set_optimizer(opt.learning_rate_stream,
                              opt.momentum,
                              opt.weight_decay,
                              model,
                              criterion=criterion)

    # init memory
    mem = Memory(opt)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    tb_logger.configure("tb_test", flush_secs=5)
    tb_logger.log_value('test_property',1,0)
  
     
    opt.stats = {
        "times_cls":{},
        "times_mem_cls":{},
        "acc_knn_training_set":{},
        "acc_val_set":{},
        "spectral_acc":{},
        "(TP+TN)/N":{},
        "precision":{},
        "F-Measure":{},
        "kappa":{},
        "var":{},
        "balanced_accuracy":{},
        "forget":[{"delta":0.0,"idx":0} for i in range(10)]
        }
    
     
    set_constant_learning_rate(opt.learning_rate_stream, optimizer)

    
    train(task_loaders, test_loader, knntrain_loader,
          train_transform, model, criterions, optimizer,
          opt, mem, logger)
        

if __name__ == '__main__':
    main()
