
from typing import Any, Dict, List

import torch

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
import pandas as pd
from immuno_glycowork import *
from torch_geometric.nn import TopKPooling, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
# %matplotlib inline
import seaborn as sns
# from google.colab import drive
# drive.mount('/content/drive')
import sys, os
from collections import Counter
import itertools
from itertools import compress
import operator
import pickle
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import warnings
from pathlib import Path

# Ignore all warnings
warnings.filterwarnings("ignore")



# df = pd.read_csv('data/glyco_targets_species_seq_all_V3.csv')

data = pd.read_csv('../data/immuno_iupac_to_smiles.csv', sep=",", quotechar='"')
print(data)
#print(df.columns)

# def preprocess_data(data, rank: str):
#     '''
#     this function removes those glycan for which rank values are not given
#     data: input dataframe
#     rank: taxonomic level'''
#     print(data.shape)
#     data = data[(data[rank] != '[]') & (data[rank] != '') & (data[rank] != ' ')]
#     print(data.shape)
#     return data

# # print(df.columns)
# data = preprocess_data(data, 'Species')
# print(data)
data.rename(columns={'glycan': 'target'}, inplace=True)
print(data)

lib = get_lib(data.target.values.tolist())
lib_size = len(lib)

# print(lib)
# print(lib_size)

train_x, val_x, train_y, val_y, id_val, class_list, class_converter = hierarchy_filter(data, rank = 'immunogenicity', 
                                                                                         min_seq = 5, wildcard_seed = False, 
                                                                                         wildcard_list = None,
                                                                                          wildcard_name = None, r = 0.1)

# print('class list is: ', class_list)
# train_x, val_x, train_y, val_y, id_val, class_list, class_converter = hierarchy_filter(df,
                                                                                    #    rank = 'Species')
# print(train_x, train_y)
glycan_graphs_train = dataset_to_graphs(train_x, train_y, libr = lib, error_catch = True)
glycan_graphs_val = dataset_to_graphs(val_x, val_y, libr = lib)

import numpy as np
import random
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) 

train_loader = DataLoader(glycan_graphs_train, batch_size = 32, shuffle = True)
val_loader = DataLoader(glycan_graphs_val, batch_size = 32, shuffle = False)
dataloaders = {'train':train_loader, 'val':val_loader}

print(train_loader)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SweetNet(torch.nn.Module):
    def __init__(self, num_classes = 1):
        super(SweetNet, self).__init__() 

        self.conv1 = GraphConv(128, 128)
        self.pool1 = TopKPooling(128, ratio = 0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio = 0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio = 0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings = lib_size+1, embedding_dim = 128)
        self.lin1 = torch.nn.Linear(256, 1024)
        self.lin2 = torch.nn.Linear(1024, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()      
  
    def forward(self, x, edge_index, batch, inference = False):
        att = 0
        x = self.item_embedding(x)
        x = x.squeeze(1) 
        # print('x is: ', x.shape)

        x = F.leaky_relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _= self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = F.leaky_relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = F.leaky_relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = x1 + x2 + x3
        
        x = self.lin1(x)
        x = self.bn1(self.act1(x))
        x = self.lin2(x)
        x = self.bn2(self.act2(x))      
        x = F.dropout(x, p = 0.5, training = self.training)

        x = self.lin3(x).squeeze(1)

        if inference:
          x_out = x1 + x2 + x3
          return x, x_out, att
        else:
          return x
        

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience = 7, verbose = False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss





def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 100.0
  best_acc = 0
  val_losses = []
  val_acc = []
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-'*10)
    
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()
        
      running_loss = []
      running_acc = []
      running_mcc = []
      for data in dataloaders[phase]:
        x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        x = x.to(device)
        # print('x is: ', x.shape)
        y = y.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          pred = model(x, edge_index, batch)
          loss = criterion(pred, y)

          if phase == 'train':
            loss.backward()
            optimizer.step()
            
        running_loss.append(loss.item())
        pred2 = np.argmax(pred.cpu().detach().numpy(), axis = 1)
        running_acc.append(accuracy_score(
                                   y.cpu().detach().numpy().astype(int), pred2))
        running_mcc.append(matthews_corrcoef(y.detach().cpu().numpy(), pred2))
        
      epoch_loss = np.mean(running_loss)
      epoch_acc = np.mean(running_acc)
      epoch_mcc = np.mean(running_mcc)
      print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(
          phase, epoch_loss, epoch_acc, epoch_mcc))
      
      if phase == 'val' and epoch_loss <= best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
      if phase == 'val':
        val_losses.append(epoch_loss)
        val_acc.append(epoch_acc)
        early_stopping(epoch_loss, model)

      scheduler.step()
        
    if early_stopping.early_stop:
      print("Early stopping")
      break
    print()
    
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))
  model.load_state_dict(best_model_wts)

  ## plot loss & accuracy score over the course of training 
  fig, ax = plt.subplots(nrows = 2, ncols = 1) 
  plt.subplot(2, 1, 1)
  plt.plot(range(epoch+1), val_losses)
  plt.title('Training of SweetNet')
  plt.ylabel('Validation Loss')
  plt.legend(['Validation Loss'],loc = 'best')

  plt.subplot(2, 1, 2)
  plt.plot(range(epoch+1), val_acc)
  plt.ylabel('Validation Accuracy')
  plt.xlabel('Number of Epochs')
  plt.legend(['Validation Accuracy'], loc = 'best')
  return model




def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.sparse_(m.weight, sparsity = 0.1)

model = SweetNet(num_classes = len(class_list))
num_params = sum(p.numel() for p in model.parameters())
print(f"number of parameters are: {num_params}")
model.apply(init_weights)
model.to(device)

early_stopping = EarlyStopping(patience = 50, verbose = True)
optimizer_ft = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay = 0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, 50)
criterion = torch.nn.CrossEntropyLoss().to(device)
model_ft = train_model(model, criterion, optimizer_ft, scheduler,
                  num_epochs = 100)