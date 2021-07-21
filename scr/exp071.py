import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import pickle

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW, Optimizer
import torchvision.models as models
from torchvision import transforms as T
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import timm

from torch.cuda.amp import autocast, GradScaler

import lightly

import swin_transfomer

import warnings 
warnings.filterwarnings('ignore')


NUM_EXP = os.path.basename(__file__)[:-3]    # expXXX(ファイル名)を取得

INPUT_DIR = '../input/dataset_atmaCup11/'
OUTPUT_DIR =  f'../exp/{NUM_EXP}/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


train_df = pd.read_csv(INPUT_DIR + 'train.csv')
test_df = pd.read_csv(INPUT_DIR + 'test.csv')

materials_df = pd.read_csv(INPUT_DIR + 'materials.csv')
techniques_df = pd.read_csv(INPUT_DIR + 'techniques.csv')

train_df['image'] = train_df['object_id'].apply(lambda x: INPUT_DIR + 'photos/' + x + '.jpg')
test_df['image'] = test_df['object_id'].apply(lambda x: INPUT_DIR + 'photos/' + x + '.jpg')



# ============================
# CFG
# ============================

class CFG:
    apex=False  # always False
    debug=False
    print_freq=100
    num_workers=8
    num_gpu = torch.cuda.device_count()
    model_name = 'swin_tiny_patch4_window7_224'
    size=224
    scheduler='CosineAnnealingWarmRestarts' # 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', GradualWarmupSchedulerV2
    epochs=250
    multi_gpu = False 
    batch_size=32  # batch size per GPUs


    # GradualWarmupSchedulerV2
    scheduler_params = {'lr_start': 7.5e-6, 'min_lr': 1e-6, 'lr_max': 1e-3 * batch_size/ 32 }
    multiplier = scheduler_params['lr_max'] / scheduler_params['lr_start']
    warmup_epochs = 3
    cosine_epochs = epochs - warmup_epochs

    pct_start = 0.1  # OneCycleLR
    div_factor = 1e3  # OneCycleLR

    min_lr=scheduler_params['min_lr']

    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    # T_max=epochs # CosineAnnealingLR
    T_0=5 # CosineAnnealingWarmRestarts
    optimizer = "Ranger"  # Adam, SGD, AdamW, Ranger,
    use_sam = False
    optimizer_params = {'lr': scheduler_params['lr_max'],
                        'weight_decay':1e-6,
                        # 'momentum': 0.9,  # SGD
                        }
    # lr=1e-4

    # augmentations
    augmentations = {
        'RandomResizedCrop': {'height': size, 'width': size},
        'Transpose': {'p': 0.5},
        'VerticalFlip': {'p': 0.5},
        'HorizontalFlip': {'p': 0.5},

        'ShiftScaleRotate': {'p': 0.5},
        'HueSaturationValue': {'hue_shift_limit': 0.2, 'sat_shift_limit': 0.2, 'val_shift_limit': 0.2, 'p': 0.5},
        # 'RandomBrightnessContrast': {'brightness_limit': (-0.1, 0.1), 'contrast_limit': (-0.1, 0.1), 'p': 0.5},
        'ToGray': {'p': 0.2},
        # 'ToSepia': {'p': 0.2},
    }
    use_course_dropout = False
    use_cutout = False

    use_mixup = False
    alpha = 1.0

    # loss function
    # criterion = 'MSELoss'  # MSELoss, CrossEntroyLoss

    # weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42 
    # if criterion == 'CrossEntropyLoss':
    #     target_size = 4
    # else:
    #     target_size = 1
    
    # training target
    training_type = 'A'  # A, B, C
    half_century = False  # targetを半世紀ごとに分割, BかCでしか効果ない

    if training_type == 'A':
        criterion = 'MSELoss'
        train_target = 'target'
        target_type = np.float32
        
    elif training_type == 'B':
        criterion = 'MSELoss'
        train_target = 'sorting_date'
        target_type = np.float32

    elif training_type == 'C':
        criterion = 'CrossEntropyLoss'
        train_target = 'target'
        target_type = np.int64

    # pretraining by materials.csv or techniques.csv
    pretraining = False
    pretrain_type = 'both'  # materials or techniques or both
    if pretrain_type == 'materials':
        pretrain_target = ['cardboard', 'chalk', 'deck paint', 'gouache (paint)', 'graphite (mineral)', 'ink', 'oil paint (paint)', 'paint (coating)', 'paper', 'parchment (animal material)', 'pencil', 'prepared paper', 'tracing paper', 'watercolor (paint)']
        pretrain_target_type = np.float32
        pretrain_criterion = 'BCEWithLogitsLoss'
    elif pretrain_type == 'techniques':
        pretrain_target = ['brush', 'counterproof', 'pen']
        pretrain_target_type = np.float32
        pretrain_criterion = 'BCEWithLogitsLoss'
    elif pretrain_type == 'both':
        pretrain_target = ['cardboard', 'chalk', 'deck paint', 'gouache (paint)', 'graphite (mineral)', 'ink', 'oil paint (paint)', 'paint (coating)', 'paper', 'parchment (animal material)', 'pencil', 'prepared paper', 'tracing paper', 'watercolor (paint)']
        pretrain_target +=  ['brush', 'counterproof', 'pen']
        pretrain_target_type = np.float32
        pretrain_criterion = 'BCEWithLogitsLoss'

    target_col = 'target'

    # self supervised
    self_supervised = False
    self_supervised_method = 'SimSiam'
    pred_hidden_dim = 512
    out_dim = 512
    num_mlp_layers = 2
    ssl_batch_size = 32

    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
if CFG.debug:
    CFG.epochs = 1
    train_df = train_df.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)



######################################
# Utils
######################################
def get_score(y_true, y_pred):
    score = mean_squared_error(y_true, y_pred)
    return np.sqrt(score)

def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # False for Faster training
    torch.backends.cudnn.banchmark = False  # True for faster training

seed_torch(seed=CFG.seed)

##############################
# Preprocessing
###############################

# https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def preprocess(train_df, materials_df, techniques_df):
    # 余分な列を切り落として、onehotにして、train_dfにマージする
    def merge(train_df, df):
        # 重複する行があったので、削除
        df = df[~df.duplicated()].reset_index()

        df_oh = pd.get_dummies(df, columns=['name'])
        df_oh = df_oh.set_index('object_id')
        df_oh = df_oh.sum(level=0).reset_index()

        # 余分な列名削除
        new_name = {}
        for name in df_oh:
            new_name[name] = name.replace('name_', '')

        df_oh = df_oh.rename(columns=new_name)

        df_cols = [c for c in df_oh.loc[:, [True, *(df_oh[[col for col in df_oh.columns if col != 'object_id']].values.sum(axis=0) > 5)]].columns if c != 'object_id']
        # print(materials_cols)
        df_oh_ = df_oh.loc[:, ['object_id', *df_cols]]

        train_df_merged = pd.merge(train_df, df_oh_, on='object_id', how='left').fillna(0.0)
        return train_df_merged
    df_merged = merge(train_df, materials_df)
    df_merged = merge(df_merged, techniques_df)
    return df_merged

for fold, (train_idx, val_idx) in enumerate(stratified_group_k_fold(train_df['image'], train_df['target'], groups=train_df['art_series_id'], k=CFG.n_fold, seed=CFG.seed)):
    train_df.loc[val_idx, 'fold'] = int(fold)

train_df['fold'] = train_df['fold'].astype(int)
train_df = preprocess(train_df, materials_df, techniques_df)
train_df.groupby(['fold', 'target']).size()

######################################
# Dataset
#######################################
def get_image(file_path):
    image = cv2.imread(file_path)[:, :, ::-1]
    # image = image.astype(np.float32)
    # image = np.vstack(image).transpose((1, 0))
    return image

class ImageDataset(Dataset):
    def __init__(self, file_names, labels=None, transform=None, return_dict=False, target_type=np.float32):
        """
        Args:
            file_names
            labels
            transform
            return_dict  (bool): if True, __getitem__ returns dict type, else, returns tuple
        """
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        self.return_dict = return_dict
        self.target_type = target_type
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = get_image(file_path)
        if self.transform:
            image = self.transform(image=image)['image']
        if self.labels is None:
            if self.return_dict:
                return {'image': image}
            else:
                return image
            
        else:
            label = np.array(self.labels[idx]).astype(self.target_type)
            if self.return_dict:
                return {'image': image, 'label': label}
            else:
                return image, label


##################################3
# Transforms
####################################

def get_transforms(*, data):
    dropout = []
    if CFG.use_course_dropout:
        dropout.append(A.CoarseDropout(p=0.5))
    if CFG.use_cutout:
        dropout.append( A.Cutout(p=0.5))
    
    if data == 'train':
        return A.Compose([
            # A.Resize(CFG.size, CFG.size),
            *(getattr(A, aug)(**arg) for aug, arg in CFG.augmentations.items()),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            *dropout,
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


################################
# Optimizer
################################3

#credit : https://github.com/Yonghongwei/Gradient-Centralization

def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,                       # lr
                 alpha=0.5, k=5, N_sma_threshhold=5,           # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False, gc_loc=True
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        # level of gradient centralization
        #self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_conv_only == False):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_conv_only == True):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        #loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                # if grad.dim() > self.gc_gradient_threshold:
                #    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # if group['weight_decay'] != 0:
                #    p_data_fp32.add_(-group['weight_decay']
                #                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if group['weight_decay'] != 0:
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])
                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])
                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

###############################
# Model
###############################

class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False, backbone=None, target_size=1):
        super().__init__()
        self.cfg = cfg

        if 'swin' in self.cfg.model_name and 'tiny' in self.cfg.model_name:
            self.model = swin_transfomer.SwinTransformer(
                img_size=CFG.size,
                in_chans=3,
                num_classes=1,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate= 0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                ape=False,
                patch_norm=True,
                use_dense_prediction=False,
            )
            self.n_features = self.model.head.in_features
            self.model.head = nn.Identity()
            state_dict = torch.load('../esvit_output/checkpoint.pth', map_location='cpu')['student']
            state_dict = fix_model_state_dict(state_dict)
            self.model.load_state_dict(
                state_dict,
            )
            for param in self.model.parameters():
                param.requires_grad = False
        else:

            self.model = timm.create_model(self.cfg.model_name, pretrained=False)
            
        if 'efficientnet' in self.cfg.model_name:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif 'resnet' in self.cfg.model_name:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif 'nfnet' in self.cfg.model_name:
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
        elif ('vit' in self.cfg.model_name):
            self.n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        
        if backbone is not None:
            self.model.load_state_dict(backbone.state_dict())

        self.fc = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(self.n_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, target_size),
            )
        # self.n_features = self.model.head.fc.in_features
        # self.model.head.fc = nn.Linear(self.n_features, self.cfg.target_size)

    def forward(self, x):
        x = self.model(x)
        output = self.fc(x)
        return output

##########################
# Scheduler
#########################3

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


################################3
# Mixup
################################

def mixup_data(x, y, alpha=1.0,
    # use_cuda=True, device="cpu"
    ):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        # lam = min(lam, 1-lam)
    else:
        lam = 1.
    batch_size = x.size()[0]

    # if use_cuda:
    #     index = torch.randperm(batch_size).to(device)
    # else:
    #     index = torch.randperm(batch_size)
    index = torch.randperm(batch_size).to(x.device)

    ## SYM
    # mixed_x = lam * x + (1 - lam) * x[index,:]
    # mixed_y = (1 - lam) * x + lam * x[index,:]
    # mixed_image  = torch.cat([mixed_x,mixed_y], 0)
    # y_a, y_b = y, y[index]
    # mixed_label  = torch.cat([y_a,y_b], 0)


    ## Reduce batch size
    # new_batch_size = batch_size // 2
    # x_i = x[ : new_batch_size]
    # x_j = x[new_batch_size : ]
    # y_a = y[ : new_batch_size]
    # y_b = y[new_batch_size : ]
    # mixed_x = lam * x_i + (1 - lam) * x_j


    ## NO SYM
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    ## Only Alpha
    # mixed_x = 0.5 * x + (1 - 0.5) * x[index,:]
    # mixed_image  = mixed_x
    # y_a, y_b = y, y[index]
    # ind_label = torch.randint_like(y, 0,2)
    # mixed_label  = ind_label * y_a + (1-ind_label) * y_b

    ## Reduce batch size and SYM
    # new_batch_size = batch_size // 2
    # x_i = x[ : new_batch_size]
    # x_j = x[new_batch_size : ]
    # y_a = y[ : new_batch_size]
    # y_b = y[new_batch_size : ]
    # mixed_x = lam * x_i + (1 - lam) * x_j
    # mixed_y = (1 - lam) * x_i + lam * x_j
    # mixed_x  = torch.cat([mixed_x,mixed_y], 0)
    # y_b = torch.cat([y_b,y_a], 0)
    # y_a = y


    # return mixed_image, mixed_label, lam
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    # sigmoid = 1.0/(1 + math.exp( 5 - 10*lam))
    # sigmoid = 4.67840515/(5.85074311 + math.exp(6.9-10.2120858*lam))
    # sigmoid = 1.531 /(1.71822 + math.exp(6.9-12.2836*lam))
    # return lambda criterion, pred: sigmoid * criterion(pred, y_a) + (1 - sigmoid) * criterion(pred, y_b)

    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


###############################3
# Helper functions
################################

from collections import OrderedDict
def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.head') or name.startswith('head'):
            continue
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
        if name.startswith('module.head') or name.startswith('head'):
            pass
    return new_state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, rank=None, world_size=None):
    if CFG.apex:
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.fc.train()
    start = end = time.time()
    global_step = 0

    input_list, output_list, lams = [], [], []

    for step, (images, labels) in enumerate(train_loader):
        batch_size = labels.size(0)
        # measure data loading time
        data_time.update(time.time() - end)
        if CFG.multi_gpu:
            images = images.to(rank)
            labels = labels.to(rank)
        else:
            images = images.to(device)
            labels = labels.to(device)
            if not isinstance(criterion, nn.CrossEntropyLoss):
                labels = labels.reshape(batch_size, -1)


        

        if CFG.use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=CFG.alpha)


        if CFG.apex:
            with autocast():
                y_preds = model(images)
                if CFG.use_mixup:
                    loss_fn = mixup_criterion(labels_a, labels_b, lam)
                    loss = loss_fn(criterion, y_preds)  # view
                else:
                    loss = criterion(y_preds, labels)  # view
        else:
            y_preds = model(images)
            if CFG.use_mixup:
                loss_fn = mixup_criterion(labels_a, labels_b, lam)
                loss = loss_fn(criterion, y_preds)  # view
            else:
                loss = criterion(y_preds, labels)  # view
        # record loss
        if (CFG.use_sam) and (CFG.gradient_accumulation_steps>1):
            input_list.append(images)
            
            if CFG.use_mixup:
                output_list.append([labels_a, labels_b])
                lams.append(lam)
            else:
                output_list.append(labels)

        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex and (not CFG.use_sam):
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if CFG.apex:
                if CFG.use_sam:
                    optimizer.first_step(zero_grad=True)
                    # scaler.step(optimizer.first_step)
                    # scaler.update()
                    for i in range(len(input_list)):
                        with autocast():
                            pred =  model(input_list[i])
                        if CFG.use_mixup:
                            loss_fn = mixup_criterion(output_list[i][0], output_list[i][1], lams[i])
                            loss = loss_fn(criterion, y_preds)  # view
                        else:
                            loss = criterion(pred, output_list[i])
                        loss = loss / CFG.gradient_accumulation_steps
                        # scaler.scale(loss).backward()
                        loss.backward()
                    optimizer.second_step(zero_grad=True)
                    # scaler.step(optimizer.second_step)
                    # scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if CFG.use_sam:
                    optimizer.first_step(zero_grad=True)
                    y_preds = model(images)
                    if CFG.use_mixup:
                        loss_fn = mixup_criterion(labels_a, labels_b, lam)
                        loss = loss_fn(criterion, y_preds)  # view
                    else:
                        loss = criterion(y_preds, labels)  # view
                    loss.backward()
                    # loss.backward()
                    optimizer.second_step(zero_grad=True)  
                else:
                    optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  #'LR: {lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   #lr=scheduler.get_lr()[0],
                   ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        batch_size = labels.size(0)
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        if not isinstance(criterion, nn.CrossEntropyLoss):
            labels = labels.reshape(batch_size, -1)

        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def pretrain_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, rank=None, world_size=None):
    if CFG.apex:
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0

    # input_list, output_list, lams = [], [], []
    avg_loss = 0.
    avg_output_std = 0.
    for step, ((x0, x1), _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x0 = x0.to(device)
        x1 = x1.to(device)
        batch_size = x0.size(0)
        


        with autocast(enabled=CFG.apex):
            y0, y1 = model(x0, x1)
        loss = criterion(y0, y1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        if CFG.self_supervised_method == 'SimSiam':
            output, _ = y0
        else:
            output = y0
        output = output.detach()
        output = F.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        avg_loss = w * avg_loss + (1 - w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

        

        losses.update(loss.item(), batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                #   'Grad: {grad_norm:.4f}  '
                  #'LR: {lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                #    grad_norm=grad_norm,
                   #lr=scheduler.get_lr()[0],
                   ))
    # the level of collapse is large if the standard deviation of the l2
    # normalized output is much smaller than 1 / sqrt(dim)
    collapse_level = max(0., 1 - math.sqrt(CFG.out_dim) * avg_output_std)
    # print intermediate results
    LOGGER.info(f'[Epoch {epoch:3d}] '
        f'Loss = {avg_loss:.2f} | '
        f'Collapse Level: {collapse_level:.2f} / 1.00')
    return collapse_level


############################
# Train loop
#############################

def train_loop(folds, fold, backbone=None, phase='train'):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_col].values 

    if phase == 'train':
        target = CFG.train_target
        target_type = CFG.target_type
        if CFG.training_type == 'C':
            target_size = 4
        else:
            target_size = 1
        
    else:
        target = CFG.pretrain_target
        target_type = CFG.pretrain_target_type
        target_size = len(target)

    train_target = train_folds[target].values 
    valid_target = valid_folds[target].values 


    if (CFG.training_type == 'B') and (phase == 'train'):
        train_target = (train_target - 1550) / 100
        valid_target = (valid_target - 1550) / 100
        if CFG.half_century:
            train_target = (2 * train_target).astype(np.int32) / 2 
            valid_target = (2 * valid_target).astype(np.int32) / 2 
    elif (CFG.training_type == 'A') and (phase == 'train'):
        train_target = train_target - 1.5
        valid_target = valid_target - 1.5
    elif CFG.training_type == 'C' and phase == 'train' and CFG.half_century:
        train_year = train_folds['sorting_date'].values / 100 - 15.5
        valid_year = valid_folds['sorting_date'].values 
        train_target = np.clip((2 * train_year).astype(np.int64), 0, 7)
        valid_target = np.clip((2 * valid_year).astype(np.int64), 0, 7)
        target_size = 8



    # target_type = np.int32 if CFG.criterion == 'CrossEntropyLoss' else np.float32



    train_dataset = ImageDataset(train_folds['image'].values, train_target, 
                                 transform=get_transforms(data='train'), target_type=target_type)
    valid_dataset = ImageDataset(valid_folds['image'].values, valid_target, 
                                 transform=get_transforms(data='valid'), target_type=target_type)
    if CFG.multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=rank, shuffle=True)

        train_loader = DataLoader(train_dataset, 
                                  sampler=train_sampler,
                                  batch_size=CFG.batch_size, 
                                  shuffle=False, 
                                  num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, rank=rank)

        valid_loader = DataLoader(valid_dataset, 
                                #   sampler=valid_sampler,
                              batch_size=CFG.batch_size * 2, 
                              shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    else:
        train_loader = DataLoader(train_dataset, 
                                batch_size=CFG.batch_size, 
                                shuffle=True, 
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, 
                                batch_size=CFG.batch_size * 2, 
                                shuffle=False, 
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=False)


    def get_optimizer(model):
        if CFG.optimizer == 'Adam':
            optimizer =  Adam
        elif CFG.optimizer == 'SGD':
            optimizer = SGD
        elif CFG.optimizer == 'AdamW':
            optimizer = AdamW
        elif CFG.optimizer == 'Ranger':
            optimizer = Ranger
        else:
            LOGGER.info(f'Optimizer {CFG.optimizer} is not implementated')
        
        if CFG.use_sam:
            return SAM(param_group, optimizer, **CFG.optimizer_params)
        else:
            return optimizer(param_group, **CFG.optimizer_params)


    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler == 'GradualWarmupSchedulerV2':
            scheduler_cosine = CosineAnnealingLR(optimizer, T_max=CFG.cosine_epochs - CFG.warmup_epochs, eta_min=CFG.min_lr, last_epoch=-1)
            scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=CFG.multiplier, total_epoch=CFG.warmup_epochs, after_scheduler=scheduler_cosine)
        elif CFG.scheduler == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer, max_lr=CFG.scheduler_params['lr_max'], pct_start=CFG.pct_start, div_factor=CFG.div_factor, epochs=CFG.epochs, steps_per_epoch=math.ceil(len(train_loader)/CFG.gradient_accumulation_steps))
        else:
            LOGGER.info(f'Scheduler {CFG.scheduler} is not implementated')
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, pretrained=False, backbone=backbone, target_size=target_size)
    if CFG.multi_gpu:
        model.to(rank)
        process_group = torch.distributed.new_group([i for i in range(world_size)])
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
        model = DDP(model, device_ids=[rank])
    else:
        model.to(device)

    if 'swin' in CFG.model_name:
        param_group = [
            {'params': model.model.parameters(), 'lr': CFG.optimizer_params['lr']/10},
            {'params': model.fc.parameters()}
        ]
    else:
        param_group = [
            {'params': model.parameters()}
        ]

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    if phase == 'train':
        loss_fn = CFG.criterion
    else:
        loss_fn = CFG.pretrain_criterion
    criterion = getattr(nn, loss_fn)()

    best_score = np.inf
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        if CFG.multi_gpu:
            train_sampler.set_epoch(epoch)
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval

        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, GradualWarmupSchedulerV2):
            scheduler.step()

        # scoring
        if phase == 'train':
            if CFG.training_type == 'C':
                preds = F.softmax(torch.tensor(preds), dim=1).numpy()
                preds_ = np.argmax(preds, axis=1)
                # preds_ = np.zeros((len(preds)))
                # for i in range(preds.shape[1]):
                #     preds_ += i*preds[:, i]
                if CFG.half_century:
                    preds_ = preds_ / 2
                preds_ = np.clip(preds_, 0, 3)
                score = get_score(valid_labels, preds_)
            elif CFG.training_type == 'A':
                preds = preds + 1.5
                preds = np.clip(preds, 0, 3)
                score = get_score(valid_labels, preds)
            else:
                preds = np.clip(preds, 0, 3)
                score = get_score(valid_labels, preds)
        else:
            score = 0
            preds = torch.sigmoid(torch.tensor(preds)).numpy()

        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        if phase == 'train':
            LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score < best_score:
            best_score = score

            torch.save({'model': model.state_dict(), 
                    'preds': preds},
                    OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_{phase}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
                
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_{phase}_best_loss.pth')

    if phase == 'train':
        if CFG.training_type == 'C':
            valid_preds = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_{phase}_best_score.pth', 
                                            map_location=torch.device('cpu'))['preds']
            
            # valid_preds_ = np.zeros((len(valid_preds)))
            # for i in range(valid_preds.shape[1]):
            #     valid_preds_ += i*valid_preds[:, i]
            valid_folds['preds'] = np.argmax( valid_preds, axis=1)
            # valid_folds['preds'] = valid_preds_

            if CFG.half_century:
                num_class = 8
                valid_folds['preds'] = np.clip(valid_folds['preds'] / 2, 0, 3)
            else:
                num_class = 4
            for i in range(num_class):
                valid_folds[f'preds_{i}'] = valid_preds[:, i]
        else:
            valid_folds['preds'] = np.clip(torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_{phase}_best_score.pth', 
                                            map_location=torch.device('cpu'))['preds'],
                                            0,
                                            3)
    else:
        saved = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_{phase}_best_loss.pth', 
                                            map_location=torch.device('cpu'))
        preds = saved['preds']
        for c, prd in zip(*(target, preds.T)):
            valid_folds[f'{c}_oof'] = prd

        backbone = model.load_state_dict(saved['model'])
        backbone = model.model
        return valid_folds, backbone


    # 
    # if CFG.multi_gpu and (rank == 0):
    #     with open(OUTPUT_DIR + f'valid_folds{fold}.pickle', 'wb') as f:
    #         pickle.dump(valid_folds, f)
    return valid_folds

###########################
# Self Supervised Learning
#############################
def pretrain_loop(train_df):
    # define the augmentations for self-supervised learning
    collate_fn = lightly.data.ImageCollateFunction(
        input_size=CFG.size,
        # require invariance to flips and rotations
        hf_prob=0.5,
        vf_prob=0.5,
        rr_prob=0.5,
        # satellite images are all taken from the same height
        # so we use only slight random cropping
        min_scale=0.5,
        # use a weak color jitter for invariance w.r.t small color changes
        cj_prob=0.2,
        cj_bright=0.1,
        cj_contrast=0.1,
        cj_hue=0.1,
        cj_sat=0.1,
    )

    # create a lightly dataset for training, since the augmentations are handled
    # by the collate function, there is no need to apply additional ones here
    dataset_train_simsiam = lightly.data.LightlyDataset(
        input_dir=INPUT_DIR + 'photos/'
    )

    # create a dataloader for training
    dataloader_train_simsiam = torch.utils.data.DataLoader(
        dataset_train_simsiam,
        batch_size=CFG.ssl_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=CFG.num_workers
    )

    # create a torchvision transformation for embedding the dataset after training
    # here, we resize the images to match the input size during training and apply
    # a normalization of the color channel based on statistics from imagenet
    test_transforms = T.Compose([
        T.Resize((CFG.size, CFG.size)),
        T.ToTensor(),
        T.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    # create a lightly dataset for embedding
    dataset_test = lightly.data.LightlyDataset(
        input_dir=INPUT_DIR + 'photos/',
        transform=test_transforms
    )
    # create a dataloader for embedding
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=CFG.ssl_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=CFG.num_workers
    )

    model = CustomModel(CFG, pretrained=False)
    num_ftrs = model.fc[1].in_features
    model.fc = nn.Identity()
    
    if CFG.self_supervised_method == 'SimSiam':
        model = lightly.models.SimSiam(
            model,
            num_ftrs=num_ftrs,
            proj_hidden_dim=CFG.pred_hidden_dim,
            pred_hidden_dim=CFG.pred_hidden_dim,
            out_dim=CFG.out_dim,
            num_mlp_layers=CFG.num_mlp_layers
        )
    else:
        model = getattr(lightly.models, CFG.self_supervised_method)(
            model,
            num_ftrs=num_ftrs,
            # proj_hidden_dim=CFG.pred_hidden_dim,
            # pred_hidden_dim=CFG.pred_hidden_dim,
            out_dim=CFG.out_dim,
            # num_mlp_layers=CFG.num_mlp_layers
        )

    model.to(device)
    if CFG.self_supervised_method == 'SimSiam':
        # SimSiam uses a symmetric negative cosine similarity loss
        criterion = lightly.loss.SymNegCosineSimilarityLoss()
    else:
        criterion = lightly.loss.NTXentLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05 * CFG.ssl_batch_size / 256,
        momentum=0.9,
        weight_decay=5e-4
    )
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler == 'GradualWarmupSchedulerV2':
            scheduler_cosine = CosineAnnealingLR(optimizer, T_max=CFG.cosine_epochs - CFG.warmup_epochs, eta_min=CFG.min_lr, last_epoch=-1)
            scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=CFG.multiplier, total_epoch=CFG.warmup_epochs, after_scheduler=scheduler_cosine)
        else:
            LOGGER.info(f'Scheduler {CFG.scheduler} is not implementated')
        return scheduler
    scheduler = None

    best_level = 1.0
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        avg_collapse_level = pretrain_fn(dataloader_train_simsiam, model, criterion, optimizer, epoch, scheduler, device)


        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, GradualWarmupSchedulerV2):
            scheduler.step()

        if avg_collapse_level < best_level:
            best_level = avg_collapse_level

            torch.save(model.backbone.model.state_dict(), 
                        OUTPUT_DIR+f'{CFG.model_name}_{CFG.self_supervised_method}_best_collapse_level.pth')
    
        
    return model.backbone.model


##############################
# Inference
############################
def inference(model, states, test_loader, device, sigmoid=False):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            if sigmoid:
                y_preds = torch.sigmoid(y_preds)
            avg_preds.append(y_preds.to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


#######################################
# Main
#######################################

def main():

    """
    Prepare: 1.train 
    """

    LOGGER.info("======= start =========")

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')
    
    if CFG.train:
        # train 
        if CFG.self_supervised:
            backbone = pretrain_loop(train_df)
        else:
            backbone = None

        

        oof_df = pd.DataFrame()
        if CFG.pretraining:
            df = pd.DataFrame()


        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                
                if CFG.pretraining:
                    LOGGER.info(f'========== fold {fold} {CFG.pretrain_type} ================')
                    _df, backbone_ = train_loop(train_df, fold, backbone=backbone, phase='pretrain')
                    df = pd.concat([df, _df])
                else:
                    backbone_ = backbone
                _oof_df = train_loop(train_df, fold, backbone=backbone_, phase='train')
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)
        if CFG.pretraining:
            df.to_csv(OUTPUT_DIR + f'{CFG.pretrain_type}_oof.csv', index=False)

        LOGGER.info('========== Inference =========')
        # ====================================================
        # inference
        # ====================================================
        if CFG.training_type == 'C':
            if CFG.half_century:
                target_size = 8
            else:
                target_size =4
        else:
            target_size = 1

        model = CustomModel(CFG, pretrained=False, target_size=target_size)
        MODEL_DIR = OUTPUT_DIR
        states = [torch.load(MODEL_DIR+f'{CFG.model_name}_fold{fold}_train_best_loss.pth') for fold in CFG.trn_fold]
        test_dataset = ImageDataset(test_df['image'].values, transform=get_transforms(data='valid'))
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                                num_workers=CFG.num_workers, pin_memory=True)
        predictions = inference(model, states, test_loader, device)

        if CFG.pretraining:
            model = CustomModel(CFG, pretrained=False, target_size=len(CFG.pretrain_target))
            states = [torch.load(MODEL_DIR+f'{CFG.model_name}_fold{fold}_pretrain_best_loss.pth') for fold in CFG.trn_fold]
            pretrain_preds = inference(model, states, test_loader, device, sigmoid=True)


        # submission
        if CFG.training_type == 'C':
            predictions = F.softmax(torch.tensor(predictions), dim=1).numpy()
            if CFG.half_century:
                num_class = 8
            else:
                num_class = 4
            for i in range(num_class):
                test_df[f'{i}'] = predictions[:, i]
            predictions = np.argmax(predictions, axis=1)
            # predictions_ = np.zeros((len(predictions)))
            # for i in range(predictions.shape[1]):
            #     predictions_ += i*predictions[:, i]
            # predictions = predictions_
            if CFG.half_century:
                predictions = predictions / 2
            # test_df.to_csv(OUTPUT_DIR + 'test_probs.csv', index=False)
        elif CFG.training_type == 'A':
            predictions = predictions + 1.5

        if CFG.pretraining:
            for i, c in enumerate(CFG.pretrain_target):
                test_df[f'{c}_pred'] = pretrain_preds[:, i] 
        
        test_df.to_csv(OUTPUT_DIR + 'test_probs.csv', index=False)


        predictions = np.clip(predictions, 0, 3)
        test_df['target'] = predictions
        test_df[['target']].to_csv(OUTPUT_DIR+'submission.csv', index=False)


if __name__ == '__main__':
    main()