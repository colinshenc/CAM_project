import time

import numpy as np
import pandas as pd
import collections
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from collections import OrderedDict
import os
import pickle
from models import *

import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch import relu, sigmoid
import torch.nn.modules.activation as activation
import matplotlib

matplotlib.use('Agg')
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn import metrics
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
import matplotlib.ticker as ticker
import copy

import h5py
# import kipoi
import json

# import seaborn as sns

############################################################
# function for loading the dataset
############################################################
# def load_datas(path_h5, batch_size):
#     data = h5py.File(path_h5, 'r')
#     dataset = {}
#     dataloaders = {}
#     # Train data
#     dataset['train'] = torch.utils.data.TensorDataset(torch.Tensor(data['train_in']),
#                                                       torch.Tensor(data['train_out']))
#     dataloaders['train'] = torch.utils.data.DataLoader(dataset['train'],
#                                                        batch_size=batch_size, shuffle=True,
#                                                        num_workers=4)
#
#     # Validation data
#     dataset['valid'] = torch.utils.data.TensorDataset(torch.Tensor(data['valid_in']),
#                                                       torch.Tensor(data['valid_out']))
#     dataloaders['valid'] = torch.utils.data.DataLoader(dataset['valid'],
#                                                        batch_size=batch_size, shuffle=True,
#                                                        num_workers=4)
#
#     # Test data
#     dataset['test'] = torch.utils.data.TensorDataset(torch.Tensor(data['test_in']),
#                                                      torch.Tensor(data['test_out']))
#     dataloaders['test'] = torch.utils.data.DataLoader(dataset['test'],
#                                                       batch_size=batch_size, shuffle=True,
#                                                       num_workers=4)
#     print('Dataset Loaded')
#     target_labels = list(data['target_labels'])
#     train_out = data['train_out']
#     return dataloaders, target_labels, train_out
#
#
# ############################################################
# # function to convert sequences to one hot encoding
# # taken from Basset github repo
# ############################################################
# def dna_one_hot(seq, seq_len=None, flatten=True):
#     if seq_len == None:
#         seq_len = len(seq)
#         seq_start = 0
#     else:
#         if seq_len <= len(seq):
#             # trim the sequence
#             seq_trim = (len(seq) - seq_len) // 2
#             seq = seq[seq_trim:seq_trim + seq_len]
#             seq_start = 0
#         else:
#             seq_start = (seq_len - len(seq)) // 2
#
#     seq = seq.upper()
#
#     seq = seq.replace('A', '0')
#     seq = seq.replace('C', '1')
#     seq = seq.replace('G', '2')
#     seq = seq.replace('T', '3')
#
#     # map nt's to a matrix 4 x len(seq) of 0's and 1's.
#     #  dtype='int8' fails for N's
#     seq_code = np.zeros((4, seq_len), dtype='float16')
#     for i in range(seq_len):
#         if i < seq_start:
#             seq_code[:, i] = 0.25
#         else:
#             try:
#                 seq_code[int(seq[i - seq_start]), i] = 1
#             except:
#                 seq_code[:, i] = 0.25
#
#     # flatten and make a column vector 1 x len(seq)
#     if flatten:
#         seq_code = seq_code.flatten()[None, :]
#
#     return seq_code
#
#
# ############################################################
# # function to train a model
# ############################################################
# def train_model(train_loader, test_loader, model, device, criterion, optimizer, num_epochs,
#                 weights_folder, name_ind, verbose):
#     total_step = len(train_loader)
#
#     train_error = []
#     test_error = []
#
#     train_fscore = []
#     test_fscore = []
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss_valid = float('inf')
#     best_epoch = 1
#
#     for epoch in range(num_epochs):
#
#         model.train()  # tell model explicitly that we train
#
#         logs = {}
#
#         running_loss = 0.0
#         running_fbeta = 0.0
#
#         for seqs, labels in train_loader:
#             x = seqs.to(device)
#             labels = labels.to(device)
#
#             # zero the existing gradients so they don't add up
#             optimizer.zero_grad()
#
#             # Forward pass
#             outputs = model(x)
#             loss = criterion(outputs, labels)
#
#             # Backward and optimize
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         # scheduler.step() #learning rate schedule
#
#         # save training loss to file
#         epoch_loss = running_loss / len(train_loader)
#         logs['train_log_loss'] = epoch_loss
#         train_error.append(epoch_loss)
#
#         # calculate test (validation) loss for epoch
#         test_loss = 0.0
#         test_fbeta = 0.0
#
#         with torch.no_grad():  # we don't train and don't save gradients here
#             model.eval()  # we set forward module to change dropout and batch normalization techniques
#             for seqs, labels in test_loader:
#                 x = seqs.to(device)
#                 y = labels.to(device)
#                 model.eval()  # we set forward module to change dropout and batch normalization techniques
#                 outputs = model(x)
#                 loss = criterion(outputs, y)
#                 test_loss += loss.item()
#
#         test_loss = test_loss / len(test_loader)
#         logs['test_log_loss'] = test_loss
#         test_error.append(test_loss)
#
#         if verbose:
#             print ('Epoch [{}], Current Train Loss: {:.5f}, Current Val Loss: {:.5f}'
#                    .format(epoch + 1, epoch_loss, test_loss))
#
#         if test_loss < best_loss_valid:
#             best_loss_valid = test_loss
#             best_epoch = epoch
#             best_model_wts = copy.deepcopy(model.state_dict())
#
#
#             model.load_state_dict(best_model_wts)
#             torch.save(best_model_wts, weights_folder + "/" + "model_epoch_" + str(best_epoch + 1) + "_" + \
#             name_ind + ".pth")  # weights_folder, name_ind
#
#             # return model, best_loss_valid
#             return model, train_error, test_error
#
#     ############################################################
#     # function to test the performance of the model
#     ############################################################
#     def run_test(model, dataloader_test, device):
#         running_outputs = []
#         running_labels = []
#         sigmoid = nn.Sigmoid()
#         with torch.no_grad():
#             for seq, lbl in dataloader_test:
#                 seq = seq.to(device)
#                 out = model(seq)
#                 out = sigmoid(out.detach().cpu())  # for BCEWithLogits
#                 running_outputs.extend(out.numpy())  # for BCEWithLogits
#                 running_labels.extend(lbl.numpy())
#         return np.array(running_labels), np.array(running_outputs)
#
#     ############################################################
#     # functions to compute the metrics
#     ############################################################
#     def compute_metrics(labels, outputs, save=None):
#         TP = np.sum(((labels == 1) * (np.round(outputs) == 1)))
#         FP = np.sum(((labels == 0) * (np.round(outputs) == 1)))
#         TN = np.sum(((labels == 0) * (np.round(outputs) == 0)))
#         FN = np.sum(((labels == 1) * (np.round(outputs) == 0)))
#         print('TP : {} FP : {} TN : {} FN : {}'.format(TP, FP, TN, FN))
#         plt.bar(['TP', 'FP', 'TN', 'FN'], [TP, FP, TN, FN])
#
#         if save:
#             plt.savefig(save)
#         else:
#             plt.show()
#
#         try:
#             print('Roc AUC Score : {:.2f}'.format(roc_auc_score(labels, outputs)))
#             print('AUPRC {:.2f}'.format(average_precision_score(labels, outputs)))
#         except ValueError:
#             pass
#         precision = TP / (TP + FP)
#         recall = TP / (TP + FN)
#         accuracy = (TP + TN) / (TP + FP + FN + TN)
#         print('Precision : {:.2f} Recall : {:.2f} Accuracy : {:.2f}'.format(precision, recall, accuracy))
#
#     #####################################################################################################
#     #####################################################################################################
#
#     ############################################################
#     # functions to compute the metrics
#     ############################################################
#     def compute_single_metrics(labels, outputs):
#         TP = np.sum(((labels == 1) * (np.round(outputs) == 1)))
#         FP = np.sum(((labels == 0) * (np.round(outputs) == 1)))
#         TN = np.sum(((labels == 0) * (np.round(outputs) == 0)))
#         FN = np.sum(((labels == 1) * (np.round(outputs) == 0)))
#
#         precision = TP / (TP + FP)
#         recall = TP / (TP + FN)
#         accuracy = (TP + TN) / (TP + FP + FN + TN)
#         mcorcoef = matthews_corrcoef(labels, np.round(outputs))
#
#         return precision, recall, accuracy, mcorcoef
#
#     ############################################################
#     # function to plot bar plot of results
#     ############################################################
#     def plot_results(labels, outputs, targets):
#
#         TP = np.sum(((labels == 1) * (np.round(outputs) == 1)), axis=0)
#         FP = np.sum(((labels == 0) * (np.round(outputs) == 1)), axis=0)
#         TN = np.sum(((labels == 0) * (np.round(outputs) == 0)), axis=0)
#         FN = np.sum(((labels == 1) * (np.round(outputs) == 0)), axis=0)
#
#         layout = go.Layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             xaxis=dict(
#                 title='Transcription factors'),
#             yaxis=dict(
#                 title='Sequences'),
#             font=dict(
#                 size=18,
#                 color='#000000'
#             ))
#
#         fig = go.Figure(data=[
#             go.Bar(name='TP', x=targets, y=TP),
#             go.Bar(name='FP', x=targets, y=FP),
#             go.Bar(name='TN', x=targets, y=TN),
#             go.Bar(name='FN', x=targets, y=FN)
#         ], layout=layout)
#         # Change the bar mode
#         fig.update_layout(barmode='stack')
#         fig.show()
#
#     ############################################################
#     # function to show training curve
#     # save - place to save the figure
#     ############################################################
#     def showPlot(points, points2, title, ylabel, save=None):
#         plt.figure()
#         fig, ax = plt.subplots()
#         # this locator puts ticks at regular intervals
#         # loc = ticker.MultipleLocator(base=0.2)
#         # ax.yaxis.set_major_locator(loc)
#         plt.plot(points)
#         plt.plot(points2)
#         plt.ylabel("Loss")
#         plt.legend(['train', 'validation'], loc='upper right')
#         plt.title(title)
#
#         if save:
#             plt.savefig(save)
#         else:
#             plt.show()



def plot():
    with open('/ubc/cs/research/shield/projects/cshen001/CAM_project/10-06-2020-23:15:19_plot_data_.txt') as file0_mp, \
    open('/ubc/cs/research/shield/projects/cshen001/CAM_project/10-04-2020-19:40:47_plot_data_.txt') as file1_mp:
    # open('/ubc/cs/research/shield/projects/cshen001/CAM_project') as file2_mp:
    # open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-04-2020-23:46:10_feat_mult_16_bs_2048_lr_2e-05_0_Maxpool_data_for_plot.txt') as file0_mp, \
    # open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-04-2020-23:46:10_feat_mult_16_bs_2048_lr_2e-05_1_Maxpool_data_for_plot.txt') as file1_mp, \
    # open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-04-2020-23:46:10_feat_mult_16_bs_2048_lr_2e-05_2_Maxpool_data_for_plot.txt') as file2_mp:
    #
    #     plot_dict_0_fm = json.load(file0_feat_mult)
    #     plot_dict_1_fm = json.load(file1_feat_mult)
    #     plot_dict_2_fm = json.load(file2_feat_mult)

        plot_dict_0_mp = json.load(file0_mp)
        plot_dict_1_mp = json.load(file1_mp)
        # plot_dict_2_mp = json.load(file2_mp)
    num_of_cnns = [10, 40, 80, 160, 200]
    feat_mult = num_of_cnns#[8,16,48,96,128,192,256,300,320]#[8,96,192,256,300,320,360,384,512]#[8, 16, 48, 96, 128, 192, 256, 300, 320, 360, 420]
    fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    #fm0 = pd.DataFrame({'feat_mult':feat_mult* 6, 'score':plot_dict_0_fm['roc_auc'] + plot_dict_0_fm['auprc'] + plot_dict_1_fm['roc_auc']+plot_dict_1_fm['auprc']+plot_dict_2_fm['roc_auc']+plot_dict_2_fm['auprc'], 'Metrics':['roc_auc0']*6 +['auprc0']*6+['roc_auc1']*6 +['auprc1']*6+['roc_auc2']*6 +['auprc2']*6})#.rename(index={0:'roc_auc',1:'auprc'},inplace=True)
    # sns.set(style="white")
    # plot_dict_0_womp = {'roc_auc':[0.80,0.83,0.84,0.85,0.85],}
    # plot_dict_1_womp =
    # plot_dict_2_womp =
    time_arr_extra_64 = [2237,7679,15985,33233,39631]
    # Plot miles per gallon against horsepower with other semantics
    # plot = sns.lineplot(x='feat_mult', y='score',
    #              alpha=1, palette="muted", hue='Metrics', marker=['*']*6+['.']*6+['v']*6+['^']*6+['p']*6+['h']*6, color=['red']*12+['blue']*12+['green']*12,
    #              data=fm0)
    # plt.plot(feat_mult, plot_dict_0_mp[4096], color='g', marker='*', label='roc_auc0', linestyle='-.',alpha=0.65)
    plt.plot(feat_mult, plot_dict_0_mp['512'], color='g', marker='.', label='bs512', alpha=0.65)
    plt.plot(feat_mult, plot_dict_1_mp['2048'], color='c', marker='v', label='bs2048', alpha=0.65)
    plt.plot(feat_mult, plot_dict_1_mp['4096'], color='m', marker='^', label='bs4096', alpha=0.65)
    plt.plot(feat_mult,time_arr_extra_64, color='y', marker='p', label='bs64', alpha=0.65)
    # plt.plot(feat_mult, plot_dict_2_mp['auprc'], color='y', marker='h', label='auprc2', linestyle=':', alpha=0.65)
    # matplotlib.rcParams['font.sans-serif'] = 'Cambria'
    #matplotlib.rcParams['font.family'] = "sans-serif"
    # plt.plot(feat_mult, plot_dict_0_fm['auprc'], color='g', marker='*', label='auprc0_wo_maxpool', linestyle='-.', alpha=0.65)
    # plt.plot(feat_mult, plot_dict_1_fm['auprc'], color='y', marker='v', label='auprc1_wo_maxpool', linestyle='-', alpha=0.65)
    # plt.plot(feat_mult, plot_dict_2_fm['auprc'], color='m', marker='p', label='auprc2_wo_maxpool', linestyle=':', alpha=0.65)

    plt.xticks(feat_mult)
    # plt.xticklabels(feat_mult)
    plt.legend(loc="lower right")
    plt.xlabel('feature multiplier')
    plt.ylabel('score')
    plt.title('Number of CNNs range results')

    plt.savefig('/ubc/cs/research/shield/projects/cshen001/CAM_project/{}.jpg'.format('final_plot'), dpi=600)
    # # columns=plot_dict_0_fm.keys())

    # print(fm0)
# def no_scheduler(*args, **kwargs):
#     return