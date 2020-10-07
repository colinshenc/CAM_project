#import plotly.graph_objects as go
import gzip
import sys
import os
import pickle
from models import ConvNetInterpPredictionMyModule

import time
# import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch import relu, sigmoid
from collections import OrderedDict
import torch.nn.modules.activation as activation
import torch.cuda
import h5py
import numpy as np
import copy
from datetime import datetime
import json
from functions import plot
def load_datas(path_h5, batch_size):
    data = h5py.File(path_h5, 'r')
    dataset = {}
    dataloaders = {}
    # Train data
    dataset['train'] = torch.utils.data.TensorDataset(torch.tensor(np.array(data['train_in']), dtype=torch.float32),
                                                      torch.tensor(np.array(data['train_out']), dtype=torch.float32),)
    dataloaders['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                       batch_size=batch_size, shuffle=True,
                                                       num_workers=6)
    print(len(dataset['train']))
    # Validation data
    dataset['valid'] = torch.utils.data.TensorDataset(torch.tensor(np.array(data['valid_in']), dtype=torch.float32),
                                                      torch.tensor(np.array(data['valid_out']), dtype=torch.float32),)
    dataloaders['valid'] = torch.utils.data.DataLoader(dataset['valid'],
                                                       batch_size=batch_size, shuffle=True,
                                                       num_workers=6)
    print(len(dataset['valid']))

    # Test data
    dataset['test'] = torch.utils.data.TensorDataset(torch.tensor(np.array(data['test_in']), dtype=torch.float32),
                                                     torch.tensor(np.array(data['test_out']), dtype=torch.float32),)
    print(len(dataset['test']))

    dataloaders['test'] = torch.utils.data.DataLoader(dataset['test'],
                                                      batch_size=batch_size, shuffle=True,
                                                      num_workers=6)
    print('Dataset Loaded')
    target_labels = list(data['target_labels'])
    train_out = data['train_out']
    return dataloaders, target_labels, train_out


def train_model(train_loader, test_loader, model, device, criterion, optimizer, num_epochs,
               weights_folder, name_ind, verbose, time):
    print('threads {}'.format(torch.get_num_threads()))
    total_step = len(train_loader)
    
    train_error = []
    test_error = []
    
    train_fscore = []
    test_fscore = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_valid = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        # start = time.time()
        model.train()

        logs = {}

        running_loss = 0.0
        running_fbeta = 0.0

        for seqs, labels in train_loader:
            x = seqs.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            #to clip the weights (constrain them to be non-negative)
            model.final.weight.data.clamp_(0)

            loss = loss.detach()
            running_loss = running_loss + loss.item()

        #save training loss to file
        epoch_loss = running_loss / len(train_loader)
        logs['train_log_loss'] = epoch_loss
        train_error.append(epoch_loss)

        #calculate test (validation) loss for epoch
        test_loss = 0.0
        test_fbeta = 0.0

        with torch.no_grad():
            model.eval()
            for seqs, labels in test_loader:
                x = seqs.to(device)
                y = labels.to(device)
                model.eval()
                outputs = model(x)
                loss_ = criterion(outputs, y)
                loss_ = loss_.detach()
                test_loss = test_loss + loss_.item()


        test_loss = test_loss / len(test_loader) #len(test_loader.dataset)
        logs['test_log_loss'] = test_loss
        test_error.append(test_loss)

        if verbose:
            print('Epoch [{}], Current Train Loss: {:.5f}, Current Val Loss: {:.5f}'.format(epoch, epoch_loss, test_loss))

        if test_loss < best_loss_valid:
            best_loss_valid = test_loss
            # best_epoch = epoch
            # best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weights_folder + "/" + "model_epoch_" + str(epoch)+"_" + name_ind + ".pth") #weights_folder, name_indd


        # model_wts = copy.deepcopy(model.state_dict())
        #     torch.save(best_model_wts, weights_folder + "/"+"model_epoch_"+str(best_epoch+1)+"_"+
        #            name_ind+".pth") #weights_folder, name_indd
        # del best_model_wts
        # end.record()
        # torch.cuda.synchronize()
        # print('EPOCH {}, training time elapsed {} milliseconds'.format(epoch, start.elapsed_time(end)))
        # end = time.time()
        with open('./{}_time.txt'.format(time), 'a+') as file:
            # file.write('data loading time elapsed {}\n\n'.format(end_load - start_load))
            # file.write('EPOCH {}, training time elapsed {} milliseconds\n'.format(epoch, start.elapsed_time(end)))
            time_interval = datetime.now()
            time_interval = time_interval.strftime("%m-%d-%Y-%H:%M:%S")
            file.write('EPOCH: {}'.format(epoch))
            file.write(' TIME: {}\n'.format(time_interval))
            # file.write('EPOCH {}, training time elapsed {} milliseconds\n'.format(epoch, end - start))

    #return model, best_loss_valid
        # del start
        # del end
    return model, train_error, test_error
if __name__ == "__main__":
    # plot()

    TIME = datetime.now()
    TIME = TIME.strftime("%m-%d-%Y-%H:%M:%S")
    # start_ = time.time()
    num_of_cnns = [10,40,80,160,200]
    batch_size = [64,512,2048,4096,] #100
    learning_rate = 0.003
    # with open('./BS_{}_NUMCNN_{}_time_.txt'.format(batch_size, num_of_cnns), 'a+') as file:
    #     # file.write('\n\n\n\n\n\n\data loading time00 elapsed {}\n\n'.format(end_load - start_load))
    #     # file.write('TOTAL TIME ELAPSED {} seconds\n\n\n\n\n'.format(end_ - start_))
    #     file.write('NUM_OF_CNNS: {}, BATCH_SIZE: {}'.format(num_of_cnns, batch_size))
    plot_dict = {}
    for _ in batch_size:
        plot_dict[_] = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA IS AVAILABLE: {}\n'.format(torch.cuda.is_available()))
    for i in batch_size:
        for j in num_of_cnns:
            with open('./{}_time.txt'.format(TIME), 'a+') as file:
                # file.write('\n\n\n\n\n\n\data loading time00 elapsed {}\n\n'.format(end_load - start_load))
                # file.write('TOTAL TIME ELAPSED {} seconds\n\n\n\n\n'.format(end_ - start_))
                file.write('BATCH_SIZE: {}, NUM_CNNS: {}\n'.format(i, j))

            start_ = time.time()
            dataloaders, target_labels, train_out = load_datas("./data/TF_binding_data.h5", i)

            target_labels = [_.decode("utf-8") for _ in target_labels]

            num_classes = 3
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)

            # start.record()
            model = ConvNetInterpPredictionMyModule(num_of_cnns=j, num_classes=num_classes).to(device)

            criterion = nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            num_epochs = 25

            #create the folder first with os.mkdir
            model, train_error, test_error = train_model(dataloaders['train'],
                                                         dataloaders['valid'], model,
                                                         device, criterion,  optimizer,
                                                         num_epochs,
                                                         "./weights_all_pos_const_20_filters_all_epochs",
                                                         "", verbose=True, time=TIME)
            end_ = time.time()
            # end_ = datetime.now()
            # end_ = end_.strftime("%m-%d-%Y-%H:%M:%S")
            plot_dict[i].append(end_ - start_)

    # end.record()
    # torch.cuda.synchronize()

    # print('data loading time elapsed {}\n\n'.format(end_load - start_load))

    # print('training time elapsed {}'.format(start.elapsed_time(end)))
    with open('./{}_plot_data_.txt'.format(TIME), 'a+') as file:
        file.write(json.dumps(plot_dict))
        # file.write('\n\n\n\n\n\n\data loading time00 elapsed {}\n\n'.format(end_load - start_load))
        # file.write('TOTAL TIME ELAPSED {} seconds\n\n\n\n\n'.format(end_ - start_))
        # file.write('\n\n\n')
