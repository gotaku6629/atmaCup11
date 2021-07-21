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


NUM_EXP = 47

INPUT_DIR = '../input/dataset_atmaCup11/'
EXP_DIR =  f'../exp/exp{NUM_EXP:03}/'
OUTPUT_DIR = f'../exp/exp{NUM_EXP:03}/tta/'


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


train_df = pd.read_csv(EXP_DIR + 'oof_df.csv')
test_df = pd.read_csv(EXP_DIR + 'test_probs.csv')

# ============================
# CFG
# ============================

class CFG:
    apex=False  # always False
    debug=False
    print_freq=100
    num_workers=4
    num_gpu = torch.cuda.device_count()
    model_name = 'resnet18d'
    size=224
    # scheduler='CosineAnnealingWarmRestarts' # 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', GradualWarmupSchedulerV2
    # epochs=25
    # multi_gpu = False 
    batch_size=16  # batch size per GPUs


    # GradualWarmupSchedulerV2
    # scheduler_params = {'lr_start': 7.5e-6, 'min_lr': 1e-6, 'lr_max': 1e-3 * batch_size/ 32 }
    # multiplier = scheduler_params['lr_max'] / scheduler_params['lr_start']
    # warmup_epochs = 3
    # cosine_epochs = epochs - warmup_epochs

    # pct_start = 0.1  # OneCycleLR
    # div_factor = 1e3  # OneCycleLR

    # min_lr=scheduler_params['min_lr']

    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    # T_max=epochs # CosineAnnealingLR
    # T_0=5 # CosineAnnealingWarmRestarts
    # optimizer = "Ranger"  # Adam, SGD, AdamW, Ranger,
    # use_sam = False
    # optimizer_params = {'lr': scheduler_params['lr_max'],
    #                     'weight_decay':1e-6,
    #                     # 'momentum': 0.9,  # SGD
    #                     }
    # lr=1e-4

    # augmentations
    # augmentations = {
    #     'RandomResizedCrop': {'height': size, 'width': size},
    #     'Transpose': {'p': 0.5},
    #     'VerticalFlip': {'p': 0.5},
    #     'HorizontalFlip': {'p': 0.5},

    #     'ShiftScaleRotate': {'p': 0.5},
    #     'HueSaturationValue': {'hue_shift_limit': 0.2, 'sat_shift_limit': 0.2, 'val_shift_limit': 0.2, 'p': 0.5},
    #     # 'RandomBrightnessContrast': {'brightness_limit': (-0.1, 0.1), 'contrast_limit': (-0.1, 0.1), 'p': 0.5},
    #     'ToGray': {'p': 0.2},
    #     # 'ToSepia': {'p': 0.2},
    # }
    use_course_dropout = False
    use_cutout = False

    # use_mixup = False
    # alpha = 1.0

    # # loss function
    # # criterion = 'MSELoss'  # MSELoss, CrossEntroyLoss

    # # weight_decay=1e-6
    gradient_accumulation_steps=1
    # max_grad_norm=1000
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
    pretraining = True
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
    # self_supervised = False
    # self_supervised_method = 'SimCLR'
    # pred_hidden_dim = 512
    # out_dim = 512
    # num_mlp_layers = 2
    # ssl_batch_size = 128

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
            nn.Dropout(0.3),
            nn.Linear(self.n_features, target_size)
            )
        # self.n_features = self.model.head.fc.in_features
        # self.model.head.fc = nn.Linear(self.n_features, self.cfg.target_size)

    def forward(self, x):
        x = self.model(x)
        output = self.fc(x)
        return output
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


def center_crop(x, pct_crop=0.1):
    x = F.interpolate(x, size=int((1+pct_crop)*CFG.size), mode='area')
    size = x.shape[2]
    return x[:, :, size//2-CFG.size//2:size//2+CFG.size//2, size//2-CFG.size//2:size//2+CFG.size//2]

def tta(model, x, apply_fn=None):
    bs, _, height, width = x.shape
    cropped = center_crop(x, pct_crop=0.4)
    x = torch.stack([x, cropped.flip(-2)],0)
    n_tta = x.shape[0]
    x = x.view(-1, 3, height, width)

    pred = model(x)
    if apply_fn is not None:
        pred = apply_fn(pred)
    pred = pred.view(n_tta, bs, -1).mean(0)
    return pred



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
            y_preds = tta(model, images, apply_fn=None)
            # y_preds = model(images)
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



def validation_loop(folds, fold, model, phase='training'):
    # trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    # train_folds = folds.loc[trn_idx].reset_index(drop=True)
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

    # train_target = train_folds[target].values 
    valid_target = valid_folds[target].values 


    if (CFG.training_type == 'B') and (phase == 'train'):
        # train_target = (train_target - 1550) / 100
        valid_target = (valid_target - 1550) / 100
        if CFG.half_century:
            # train_target = (2 * train_target).astype(np.int32) / 2 
            valid_target = (2 * valid_target).astype(np.int32) / 2 
    elif (CFG.training_type == 'A') and (phase == 'train'):
        # train_target = train_target - 1.5
        valid_target = valid_target - 1.5
    elif CFG.training_type == 'C' and phase == 'train' and CFG.half_century:
        # train_year = train_folds['sorting_date'].values / 100 - 15.5
        valid_year = valid_folds['sorting_date'].values 
        # train_target = np.clip((2 * train_year).astype(np.int64), 0, 7)
        valid_target = np.clip((2 * valid_year).astype(np.int64), 0, 7)
        target_size = 8

    valid_dataset = ImageDataset(valid_folds['image'].values, valid_target, 
                                 transform=get_transforms(data='valid'), target_type=target_type)
    valid_loader = DataLoader(valid_dataset, 
                                batch_size=CFG.batch_size * 2, 
                                shuffle=False, 
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    model.to(device)


    if phase == 'train':
        loss_fn = CFG.criterion
    else:
        loss_fn = CFG.pretrain_criterion
    criterion = getattr(nn, loss_fn)()

    avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

    # scoring
    if phase == 'train':
        if CFG.training_type == 'C':
            preds = F.softmax(torch.tensor(preds), dim=1).numpy()
            # preds_ = np.argmax(preds, axis=1)
            preds_ = np.zeros((len(preds)))
            for i in range(preds.shape[1]):
                preds_ += i*preds[:, i]
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


    if phase == 'train':
        if CFG.training_type == 'C':
            valid_preds = preds
            
            valid_preds_ = np.zeros((len(valid_preds)))
            for i in range(valid_preds.shape[1]):
                valid_preds_ += i*valid_preds[:, i]
            # valid_folds['preds'] = np.argmax( valid_preds, axis=1)
            valid_folds['preds'] = valid_preds_

            if CFG.half_century:
                num_class = 8
                valid_folds['preds'] = np.clip(valid_folds['preds'] / 2, 0, 3)
            else:
                num_class = 4
            for i in range(num_class):
                valid_folds[f'preds_{i}'] = valid_preds[:, i]
        else:
            valid_folds['preds'] = np.clip(preds,
                                            0,
                                            3)
    else:

        for c, prd in zip(*(target, preds.T)):
            valid_folds[f'{c}_oof'] = prd
        return valid_folds

    return valid_folds
  



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
        # if CFG.self_supervised:
        #     backbone = pretrain_loop(train_df)
        # else:
        #     backbone = None

        

        oof_df = pd.DataFrame()
        if CFG.pretraining:
            df = pd.DataFrame()


        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                
                if CFG.pretraining:
                    LOGGER.info(f'========== fold {fold} {CFG.pretrain_type} ================')

                    model = CustomModel(CFG, pretrained=False, target_size=len(CFG.pretrain_target))
                    model.load_state_dict(torch.load(
                        EXP_DIR + f'{CFG.model_name}_fold{fold}_pretrain_best_loss.pth')['model']
                        )
                    _df = validation_loop(train_df, fold, model, phase='pretrain')
                    df = pd.concat([df, _df])
                
                if CFG.training_type == 'C':
                    if CFG.half_century:
                        target_size = 8
                    else:
                        target_size =4
                else:
                    target_size = 1

                model = CustomModel(CFG, pretrained=False, target_size=target_size)
                model.load_state_dict(
                    torch.load(EXP_DIR+f'{CFG.model_name}_fold{fold}_train_best_loss.pth')['model']
                    )

                _oof_df = validation_loop(train_df, fold, model, phase='train')
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
        MODEL_DIR = EXP_DIR
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
            # predictions = np.argmax(predictions, axis=1)
            predictions_ = np.zeros((len(predictions)))
            for i in range(predictions.shape[1]):
                predictions_ += i*predictions[:, i]
            predictions = predictions_
            if CFG.half_century:
                predictions = predictions / 2
            # test_df.to_csv(EXP_DIR + 'test_probs.csv', index=False)
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