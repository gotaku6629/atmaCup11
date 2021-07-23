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
from sklearn import linear_model

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

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

import warnings 
warnings.filterwarnings('ignore')

NUM_EXP = os.path.basename(__file__)[:-3]    # expXXX(ファイル名)を取得

INPUT_DIR = '../input/dataset_atmaCup11/'
OUTPUT_DIR =  f'../exp/{NUM_EXP}/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# train_df = pd.read_csv(INPUT_DIR + 'train.csv')
# test_df = pd.read_csv(INPUT_DIR + 'test.csv')

# materials_df = pd.read_csv(INPUT_DIR + 'materials.csv')
# techniques_df = pd.read_csv(INPUT_DIR + 'techniques.csv')

# train_df['image'] = train_df['object_id'].apply(lambda x: INPUT_DIR + 'photos/' + x + '.jpg')
# test_df['image'] = test_df['object_id'].apply(lambda x: INPUT_DIR + 'photos/' + x + '.jpg')


EXP_DIR = '../exp/'

class CFG:
    algo = 'Ridge'

    # params = {
    #     'objective': 'regression',
    #     'metric': 'rmse',
    #     # 'num_class': 4,
    #     'num_iterations': 10000,
    #     'learning_rate': 0.001,
    #     'num_leaves': 16,
    #     'seed': 42,
    #     'max_depth': 75,
    #     'min_data_in_leaf': 12,
    #     'bagging_fraction': 0.9,
    #     'early_stopping_round': 20,
    #     'bagging_freq': 2,
    #     'feature_fraction': 0.92,
    #     'lambda_l1': 4.5,
    #     'lambda_l2': 0.5,
    #     'verbosity': -1,
    # }
    params = {
        'alpha': 0.01,  # Ridge, Lasso
        # 'l1_ratio': 0.5,  # ElasticNet
        'random_state': 42,
        }

    exps = ['exp047', 'exp058', 'exp065', 'exp076', 'exp078']
    exps_tta = [False, False, True, True, True]
    # power = 0.4

    target_col = 'target'

    # pretrain_target = ['cardboard', 'chalk', 'deck paint', 'gouache (paint)', 'graphite (mineral)', 'ink', 'oil paint (paint)', 'paint (coating)', 'paper', 'parchment (animal material)', 'pencil', 'prepared paper', 'tracing paper', 'watercolor (paint)']
    # pretrain_target +=  ['brush', 'counterproof', 'pen']

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


def load_result_csv(num_exp, tta=False):
    if os.path.exists(EXP_DIR + f'{num_exp}/tta/') and tta:
        exp_dir = EXP_DIR + f'{num_exp}/tta/'

    else:
        exp_dir = EXP_DIR + f'{num_exp}/'
    ret = {}
    ret['submission'] = pd.read_csv(exp_dir + 'submission.csv')
    ret['oof'] = pd.read_csv(exp_dir + 'oof_df.csv')
    if os.path.exists(exp_dir + 'test_probs.csv'):
        ret['test_probs'] = pd.read_csv(exp_dir + 'test_probs.csv')
    if os.path.exists(exp_dir + 'both_oof.csv'):
        ret['both_oof'] = pd.read_csv(exp_dir + 'both_oof.csv')
    return ret

class LGBMModel():
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        train_data = lgb.Dataset(X_train, label=y_train)
        eval_data = lgb.Dataset(X_val, y_val, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=eval_data
        )
        # val_pred = self.model.predict(X_val, iteration=self.model.best_iteration)
        # score = get_score(y_val, val_pred)
        return self.model
    def predict(self, X):
        pred = self.model.predict(X, iteration=self.model.best_iteration_)
        return pred
    def save_model(self, path):
        self.model.save_model(path)
    def load_model(self, model_file):
        self.model = lgb.Booster(model_file=model_file)


def postprocess(preds, power=1/6):
    
    preds_mask0 = np.sum([(pred)**power for pred in preds.T], axis=0) < 1
    preds_mean = np.mean(preds, axis=1)
    preds_mean[preds_mask0] = 0
    
#     print(np.sum(preds_mask0))
    
    preds_mask3 = np.sum([(3-pred)**power for pred in preds.T], axis=0) < 1
    preds_mean[preds_mask3] = 3
    return preds_mean
    



def train_loop(folds, fold, feat_cols):
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_col].values

    train_feat = train_folds[feat_cols]
    valid_feat = valid_folds[feat_cols] 

    train_target = train_folds[CFG.target_col] 
    valid_target = valid_folds[CFG.target_col] 

    if CFG.algo == 'lightgbm':
        model = LGBMModel(CFG.params)
        model = model.fit(train_feat, train_target, valid_feat, valid_target)
    else:
        model = getattr(linear_model, CFG.algo)(**CFG.params)
        model = model.fit(train_feat, train_target)

    val_preds = model.predict(valid_feat)
    val_preds = np.clip(val_preds, 0, 3)
    # val_preds = np.argmax(val_preds, axis=1)
    score = get_score(valid_target, val_preds)
    LOGGER.info(f'CV: {score}')


    if CFG.algo == 'lightgbm':
        model.save_model(OUTPUT_DIR + f'lightgbm_fold{fold}.txt')
    else:
        with open(OUTPUT_DIR + f'{CFG.algo}_fold{fold}.pickle', 'wb') as f:
            pickle.dump(model, f)

    return val_preds, model

def main():
    results = [load_result_csv(exp, tta) for exp, tta in zip(*(CFG.exps, CFG.exps_tta))]

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    feat_cols = []
    train_df['target'] = results[0]['oof']['target']
    train_df['sorting_date'] = results[0]['oof']['sorting_date']
    train_df['fold'] = results[0]['oof']['fold']

    train_df['preds'] = 0
    test_df['target'] = 0

    train_preds =[]
    test_preds = []

    for exp, result in zip(*(CFG.exps, results)):
        oof = result['oof']
        sub = result['submission']
        train_df[f'pred_{exp}'] = oof['preds']
        test_df[f'pred_{exp}'] = sub['target']

        train_preds.append(oof['preds'])
        test_preds.append(sub['target'])
        
        # train_df['preds'] += oof['preds'] / len(results)
        # test_df['target'] += sub['target'] / len(results)

        feat_cols.append(f'pred_{exp}')
        # if 'both_oof' in result.keys():
        #     train_both = result['both_oof']
        #     test_both = result['test_probs']
        #     for col in CFG.pretrain_target:
        #         train_df[f'{col}_{exp}'] = train_both[f'{col}_oof']
        #         test_df[f'{col}_{exp}'] = test_both[f'{col}_pred']
        #         feat_cols.append(f'{col}_{exp}')
        # test_probs = result['test_probs']
        # # print(test_probs.columns)
        # if '0' in test_probs.columns.tolist():
        #     for i in range(4):
        #         train_df[f'probs_{i}_{exp}'] = oof[f'preds_{i}']
        #         test_df[f'probs_{i}_{exp}'] = test_probs[f'{i}']
        #         feat_cols.append(f'probs_{i}_{exp}')


    # print(train_df.columns)
    # print('features: ', feat_cols)
    
    # train
    train_df['preds'] = 0
    importance_df = pd.DataFrame()
    models = []
    for fold in range(max(train_df['fold'].values)+1):
        LOGGER.info(f'============== fold{fold} ===================')
        val_preds, model = train_loop(train_df, fold, feat_cols)
        train_df.loc[train_df['fold']==fold, 'preds'] = val_preds

        if CFG.algo == 'lightgbm':
            tmp = pd.DataFrame()
            tmp['feature'] = model.feature_name()
            tmp['importance'] = model.feature_importance(importance_type='gain')
            tmp['fold'] = fold
            importance_df = importance_df.append(tmp)

        models.append(model)
    score = get_score(train_df['target'], train_df['preds'])
    # LOGGER.info(f'CV score: {score}')
    
    if CFG.algo == 'lightgbm':
        order = list(importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False).index )

        sns.barplot(x=importance_df['importance'], y=importance_df['feature'], order=order).get_figure().savefig(OUTPUT_DIR + 'feature_importance.png')

    test_preds = []
    for model in models:
        test_pred = model.predict(test_df[feat_cols])
        # test_pred = n.argmax(test_pred, axis=1)
        test_preds.append(test_pred)
    test_preds = np.mean(test_preds, axis=0)

    # test_preds = np.clip(test_preds, 0, 3)

    # train_preds = postprocess(train_df[feat_cols].values, power=CFG.power)
    # test_preds = postprocess(test_df[feat_cols].values, power=CFG.power) 
    
    # train_df['preds'] = np.clip(train_df['preds'], 0, 3)
    test_df['target'] = np.clip(test_preds, 0, 3)
    
    LOGGER.info(CFG.exps)
    LOGGER.info(CFG.exps_tta)
    score = get_score(train_df['target'].values, train_df['preds'].values)
    LOGGER.info(f'CV score: {score}')
    
    train_df.to_csv(OUTPUT_DIR + 'oof_df.csv', index=False)
    test_df.to_csv(OUTPUT_DIR + 'test_df.csv', index=False)
    test_df['target'].to_csv(OUTPUT_DIR + 'submission.csv', index=False)
    

if __name__ == '__main__':
    main()