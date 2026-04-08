from typing import Any, Dict, List

import torch

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
import pandas as pd
from glycowork import *
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
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve



# Ignore all warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, SAGEConv, GINConv
from torch_geometric.data import Data, DataLoader
from torch.nn import Sequential as seq, ReLU, Linear as linear

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


############################

data = pd.read_csv('../data/iupac_to_smiles.csv', sep=",", quotechar='"')
rank = 'Kingdom'

data.rename(columns={'glycan': 'target'}, inplace=True)
print(data)

lib = get_lib(data.target.values.tolist())
lib_size = len(lib)

rank = 'Kingdom'
X_out, _, _, class_list, _ = hierarchy_filter(data, rank = 'Kingdom', min_seq = 5, 
                                          wildcard_seed = False, 
                                        wildcard_list = None,
                                        wildcard_name = None, r = 0.1)

print(X_out)
model_ft = SweetNet(num_classes = len(class_list))
#model_filename = f'models/best_model.pth'
model_ft.load_state_dict(torch.load('models/best_model.pth'))
model_ft = model_ft.to(device)

X = X_out.target.values.tolist()
y = X_out[rank].values.tolist()
print(type(X), len(X))
print(type(y), len(y))

#abc = list(range(len(taxonomic_glycans.species.values.tolist())))
glycan_graphs, y_list = dataset_to_graphs(X, y, libr = lib, error_catch = True)
glycan_loader = DataLoader(glycan_graphs, batch_size = 32, shuffle = False)

res = []
for data in glycan_loader:
    x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)
    model_ft = model_ft.eval()
    pred, out, _ = model_ft(x, edge_index, batch, inference = True)
    res.append(out)


res2 = [res[k].detach().cpu().numpy() for k in range(len(res))]
res2 = pd.DataFrame(np.concatenate(res2))


tsne_emb = TSNE(random_state = 42).fit_transform(res2)

print(tsne_emb, tsne_emb.shape)
print(len(y_list))
plt.figure(figsize=(9,9))
sns.scatterplot(x = tsne_emb[:,0], y = tsne_emb[:,1], s = 20, alpha = 0.4,
                hue = y_list, palette = 'colorblind', rasterized = True, legend=False)
print('scatter plot is on the way')
    # plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.xlabel('t-SNE Dim1')
plt.ylabel('t-SNE Dim2')
plt.title(f'{rank}')
plt.tight_layout()
    
plot_path = Path('embeddings')
plot_path.mkdir(parents = True, exist_ok=True)
output_path = f'./embeddings/output_{rank}_plot.png' 

plt.savefig(output_path)
print("Plot saved as 'output_{rank}_plot.png'")

