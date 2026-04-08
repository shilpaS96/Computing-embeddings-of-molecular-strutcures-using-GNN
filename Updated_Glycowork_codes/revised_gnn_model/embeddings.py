from typing import Any, Dict, List

import torch

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
import pandas as pd
from SuSNet_fun import *
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


class SuperSweetNet(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, conv_type = 'GCNConv'):
        super(SuperSweetNet, self).__init__()


        if conv_type == 'sage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        elif conv_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)

        elif conv_type == 'gin':
            self.conv1 = GINConv(seq(linear(input_dim, hidden_dim), ReLU(), linear(hidden_dim, hidden_dim)))
            self.conv2 = GINConv(seq(linear(hidden_dim, hidden_dim), ReLU(), linear(hidden_dim, hidden_dim)))
            self.conv3 = GINConv(seq(linear(hidden_dim, hidden_dim), ReLU(), linear(hidden_dim, hidden_dim)))

        elif conv_type == 'GraphConv':
            self.conv1 = GraphConv(input_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, hidden_dim)
            self.conv3 = GraphConv(hidden_dim, hidden_dim)


        # self.conv1 = GCNConv(input_dim, hidden_dim)
        '''
        input_dim: The number of input features per node. This tells the layer how many features each node 
        in the input graph has.
        hidden_dim: The number of output features per node. This determines the dimensionality of the 
        feature vectors for each node after the convolution operation.'''
        self.pool1 = TopKPooling(hidden_dim, ratio=0.8)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = TopKPooling(hidden_dim, ratio=0.8)
        # self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.pool3 = TopKPooling(hidden_dim, ratio=0.8)
        self.lin1 = linear(hidden_dim * 2, 1024)
        self.lin2 = linear(1024, 64)
        self.lin3 = linear(64, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, batch, inference = False):
        
        att = 0
        # print(x.shape, x.ndim)
        # # x = self.item_embedding(x) # 
        # print(x.shape, x.ndim)
        # # x = x.squeeze(1) 
        # print(x.shape, x.ndim)
        # x = x.reshape(-1, x.shape[0])
        # print(x.shape, x.ndim)
        # print(edge_index.max().item(), edge_index.min().item())
        # print(x.size(0))

        
        # x = x.squeeze(1)
        # print(x.shape, x.ndim)

        if edge_index.max().item() >= x.size(0) or edge_index.min().item() < 0:
            # print('an error')
            raise ValueError(f"Invalid edge_index values: {edge_index}")
        """
        try:
            #print("Before conv1:")
            #print("x:", x)
            print(x.shape)
            print(x.dtype, edge_index.dtype)
            x = x.float()
            # edge_index = edge_index.float()
            print("x shape:", x.shape)
            print("edge_index shape:", edge_index.shape)
            print("edge_index:", edge_index)
            x = self.conv1(x, edge_index)
            print('Scenario 1, conv1 executed')
            print(x.shape, x.ndim)
            #print("After conv1, before activation:")
            #print("x:", x)
            x = F.leaky_relu(x)
            print('leaky_relu is executed')
            print(x.shape, x.ndim)
            #print("After activation:")
            #print("x:", x)
        except RuntimeError as e:
            print("Error during GCNConv operation:", e)
            raise
        
        print(x.shape)
        print(x.dtype, edge_index.dtype)
        """
        x = x.float().to(device)
        edge_index = edge_index.to(device)
        # print("x shape:", x.shape, x.ndim)
        # print("edge_index shape:", edge_index.shape)
        # print("edge_index:", edge_index)
        x = F.leaky_relu(self.conv1(x, edge_index))
        '''
        During the forward pass, you need to pass the actual data to the layer to perform the 
        graph convolution operation. The GCNConv layer uses the weight matrices 
        (initialized with input_dim and hidden_dim) to transform the input features x 
        based on the connectivity information in edge_index.'''
        # print('Scenario 1, conv1 and leaky relu are executed')
        # print(x.shape, x.ndim)

        x, edge_index, _, batch, _, _= self.pool1(x, edge_index, None, batch)
        # print('pooling 1 is done')
        # print('gmp', gmp(x, batch).shape, 'gap', gap(x, batch).shape)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = F.leaky_relu(self.conv2(x, edge_index))
        # print('Scenario 2, conv2 and leaky relu are executed')
        # print(x.shape, x.ndim)
     
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pooling 2 is done')
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = F.leaky_relu(self.conv3(x, edge_index))
        # print('Scenario 3, conv3 and leaky relu are executed')
        # print(x.shape, x.ndim)

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pooling 3 is done')
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = x1 + x2 + x3
        # print('x = x1 + x2 + x3', x.shape, x.ndim)
        # print(num_classes)
        x = self.lin1(x)
        # print('lin1 is executed', x.shape, x.ndim)
        x = self.bn1(self.act1(x))
        # print('bn1 is executed', x.shape, x.ndim)
        x = self.lin2(x)
        # print('lin2 is executed', x.shape, x.ndim)
        x = self.bn2(self.act2(x))      
        # print('bn1 is executed', x.shape, x.ndim)
        x = F.dropout(x, p = 0.5, training = self.training)
        # print(x.shape, x.ndim)

        x = self.lin3(x).squeeze(1)
        # print('lin3 is executed', x.shape, x.ndim)

        if inference:
          x_out = x1 + x2 + x3
        #   print(x_out, x_out.shape, x_out.ndim)
          return x, x_out, att
        else:
        #   print("printing before returning", x.shape, x.ndim)
          return x.to(device)
        

    def get_embeddings(self, x, edge_index, batch, inference=True):
        """
        extract the final embeddings before the final classification layer"""

        x = x.float().to(device)
        edge_index = edge_index.to(device)

        x = F.leaky_relu(self.conv1(x, edge_index))
        '''
        During the forward pass, you need to pass the actual data to the layer to perform the 
        graph convolution operation. The GCNConv layer uses the weight matrices 
        (initialized with input_dim and hidden_dim) to transform the input features x 
        based on the connectivity information in edge_index.'''
        # print('Scenario 1, conv1 and leaky relu are executed', x.shape)
        # print(x.shape, x.ndim)

        x, edge_index, _, batch, _, _= self.pool1(x, edge_index, None, batch)
        # print('pooling 1 is done')
        # print('gmp', gmp(x, batch).shape, 'gap', gap(x, batch).shape)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = F.leaky_relu(self.conv2(x, edge_index))
        # print('Scenario 2, conv2 and leaky relu are executed', x.shape, x1.shape)
        # print(x.shape, x.ndim)
     
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pooling 2 is done')
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        x = F.leaky_relu(self.conv3(x, edge_index))
        # print('Scenario 3, conv3 and leaky relu are executed', x.shape, x2.shape)
        # print(x.shape, x.ndim)

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pooling 3 is done')
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)
        # print('x3', x3.shape)

        x_embeddings = x1 + x2 + x3
        # print(x_embeddings.shape)

        return x_embeddings

###################

def get_full_data_emebeddings(model, dataloader):
    embeddings_list = []
    # for phase in ['train', 'test']:
    for data in dataloader:
        x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        x = x.to(device)
        y = y.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        model = model.to(device)
        model = model.eval()

        embeddings = model.get_embeddings(x, edge_index, batch)
        embeddings_list.append(embeddings.cpu().detach().numpy())

    return embeddings_list

    ############################

data = pd.read_csv('../data/nonrectified_iupac_to_smiles.csv', sep=",", quotechar='"')
data = data[data['smiles'].apply(lambda smiles: Chem.MolFromSmiles(smiles) is not None)]
print(data.shape)


rank = 'Kingdom'

X_out, _ , _ , class_list, class_converter = hierarchy_filter(data, rank = 'Kingdom', min_seq = 5, wildcard_seed = False, wildcard_list = None,
                                                                                          wildcard_name = None, r = 0.1)
  
print(X_out.shape)
#print(len(class_list))

X = X_out['smiles'].values.tolist()
classes = X_out[rank].values.tolist()
#print(class_converter)

reverse_converter = {v: k for k, v in class_converter.items()}

# class_list = list(sorted(list(set(data[rank].values.tolist()))))
# class_converter = {class_list[k]:k for k in range(len(class_list))}
# data[rank] = [class_converter[k] for k in df[rank].values.tolist()]
X_list, y = from_smiles_list(X, classes)
print(len(y))
#print(set(y))
#print(len(X_list))

class_names = [reverse_converter[cls] for cls in y]
#print(class_names)

data_loaders = DataLoader(X_list, batch_size = 32, shuffle = True) #, drop_last=True)
input_dim = 9
hidden_dim = 128
num_classes = len(class_list)


embeddings = {}

#model = input('Which model you are using [SweetNet/SuperSweetNet]: ')

#if model = 'SuperSweetNet':
for conv_type in ['GraphConv']: #, 'gin', 'sage', 'gcn']:
    
    model_ft = SuperSweetNet(input_dim, hidden_dim, num_classes, conv_type=conv_type)
    print(conv_type)
    model_filename = f'models/before_best_{conv_type}_model.pth'
    print(model_filename)
    model_ft.load_state_dict(torch.load(model_filename))
    e = get_full_data_emebeddings(model_ft, data_loaders)
    print(len(e))
    embeddings_df = pd.DataFrame(np.concatenate(e))
    print('embeddings for: ', conv_type)
    print(embeddings_df)
    embeddings[conv_type] = embeddings_df
    # print(embeddings)

print(embeddings)
print(embeddings.keys())

# model = SuperSweetNet(input_dim, hidden_dim, num_classes, conv_type=conv_type)

for k, v in embeddings.items():
    print(k, v)
    print(v.shape)
    tsne_emb = TSNE(random_state=42).fit_transform(v)
    #print(tsne_emb, tsne_emb.shape)
    print(len(y))
    plt.figure(figsize=(9,9))
    sns.scatterplot(x = tsne_emb[:,0], y = tsne_emb[:,1], s = 20, alpha = 0.4,
                hue = class_names, palette = 'colorblind', rasterized = True)
    print('scatter plot is on the way')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='small')
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xlabel('t-SNE Dim1')
    plt.ylabel('t-SNE Dim2')
    plt.title(f'{k}_{rank}')
    plt.tight_layout()
    
    plot_path = Path('revised_embeddings')
    plot_path.mkdir(parents = True, exist_ok=True)
    output_path = f'./revised_embeddings/before_output_{rank}_{k}_plot.png' 

    plt.savefig(output_path)
    print("Plot saved as 'before_output_{rank}_{k}_plot.png'")



