# in this code file, GCNConv is used
#* III *#
# this is the main proposed SuperSweetNet model for our dataset



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

        x_embeddings = x1 + x2 + x3

        return x_embeddings



# Example usage
#model = SuperSweetNet(input_dim=9, hidden_dim=128, output_dim=128, num_classes=8, lib_size=1000)
#print(model)




class EarlyStopping:
   """Early stops the training if validation loss doesn't improve after a given patience."""
   def __init__(self, patience = 10, verbose = False):
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


## Train Model
def train_model(model, criterion, optimizer, scheduler, dataloaders, conv_type, num_epochs = 30):
    """
    model: trining model for instance GNN
    criterion: 
    optimizer:
    scheduler,
    num_epochs: number of times forward and backward pass
    source: https://github.com/BojarLab/SweetNet/blob/main/SweetNet_code.ipynb"""

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    best_acc = 0
    test_losses = []
    test_acc = []
    test_mcc = []
    # y_set = set()
    # problem = 0
    for epoch in range(num_epochs):

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                # print('phase is train')
                model.train()
            else:
                # print('phase is test')
                model.eval()

            running_loss = []
            running_acc = []
            running_mcc = []
            
            #i = 0
            for data in dataloaders[phase]:
                # print("data", data)
                # print(phase)
                # print(data)
                
                x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
                x = x.to(device)# .cuda()
                y = y.to(device) #.cuda()
                # print('x is: ', x.shape)
                # print('y is: ',y.shape)
                # i += 1
                # print('batch is: ',batch, i)
                edge_index = edge_index.to(device) #.cuda()
                batch = batch.to(device) #.cuda()
                # print((batch.tolist()))

                """ 
                for debugging
                if max(batch.tolist()) != 63:
                    # problem += 1
                    # print('problem', problem)
                    # print('list of smiles: ', len(data.smiles))
                    num_nodes = 0
                    for index, smile in enumerate(data.smiles):
                        
                        mol = Chem.MolFromSmiles(smile)
                        if mol is None:
                            print('mol is none', index, smile)
                            
                        else:
                            num_nodes += mol.GetNumAtoms()
                    print(num_nodes)
                    break
                """    
                optimizer.zero_grad() # Clears old gradients before computing new ones

                with torch.set_grad_enabled(phase == 'train'): # Enables gradient computation only during training
                    # print(x.shape)
                    pred = model(x, edge_index, batch) # Performs a forward pass to get predictions
                    loss = criterion(pred, y)

                    if phase == 'train':
                        # If in training mode, performs backpropagation and updates model weights
                        loss.backward()
                        optimizer.step()

                running_loss.append(loss.item()) # Adds the loss for the current batch to the list
                pred2 = np.argmax(pred.cpu().detach().numpy(), axis = 1) # Gets the predicted class labels by taking the argmax of predictions.
                running_acc.append(accuracy_score( # computes and stores acc
                                   y.cpu().detach().numpy().astype(int), pred2)) 
                running_mcc.append(matthews_corrcoef(y.detach().cpu().numpy(), pred2)) # computes and stores matthews corr. coef.
                pred_prob = torch.softmax(pred, dim=1)  # Predicted probabilities for multiclass AUC
                # pred_prob = pred_prob.cpu().detach().numpy()
                # break # remove this for full training
            epoch_loss = np.mean(running_loss) # Averages the loss over all batches for the epoch.
            epoch_acc = np.mean(running_acc) # Averages the acc over all batches for the epoch.
            epoch_mcc = np.mean(running_mcc) # Averages the mcc over all batches for the epoch.
            print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_mcc))

            from sklearn.preprocessing import label_binarize
            # compute precision, recall, 1-Score and AUC
            if phase == 'test':
                # print(pred_prob.shape)
                # print(y.shape)
                # Binarize the true labels for multiclass AUC calculation
                num_classes = pred_prob.shape[1]  # Number of classes
                # print(num_classes)
                y_true_binarized = label_binarize(y.cpu().detach().numpy().astype(int), classes=list(range(num_classes)))
                # print(y_true_binarized.shape)
                precision = precision_score(y.cpu().detach().numpy().astype(int), pred2, average='macro')
                # print(precision)
                recall = recall_score(y.cpu().detach().numpy().astype(int), pred2, average='macro')
                # print(recall)
                f1 = f1_score(y.cpu().detach().numpy().astype(int), pred2, average='macro')
                # print(f1)
                # print(len(np.unique(y.cpu().detach().numpy().astype(int))))
                try:
                    # pred_prob = pred_prob.shape[]
                    auc = roc_auc_score(y_true_binarized, pred_prob.cpu().detach().numpy(), multi_class='ovr', average='macro')
                    # print(auc)

                except ValueError:
                    print('error occurred')
                    auc = 0

                epoch_precision = np.mean(precision)
                epoch_recall = np.mean(recall)
                epoch_f1 = np.mean(f1)
                epoch_auc = np.mean(auc)


            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_precision = epoch_precision
                best_recall = epoch_recall
                best_f1 = epoch_f1
                best_auc = epoch_auc

            if phase == 'test':
                test_losses.append(epoch_loss)
                test_acc.append(epoch_acc)
                early_stopping(epoch_loss, model)
                test_mcc.append(epoch_mcc)

            scheduler.step()
            # break # remove this for full training
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # print()
        
    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))

    performance_metric = {'accuracy' : best_acc, 'loss' : best_loss, 'precision' : best_precision, 'recall' : best_recall, 
                          'f1-score' : best_f1, 'auc' : best_auc}
    
    return model, performance_metric, best_model_wts, test_losses, test_acc, test_mcc, epoch

    # create folder for saving models
    # model_path = Path('models')
    # model_path.mkdir(parents = True, exist_ok=True)

    # #create model save path
    # model_name = f'best_{conv_type}_model.pth'
    # model_save_path = model_path / model_name
    # model.load_state_dict(best_model_wts)

    # torch.save(best_model_wts, model_save_path)

    ## plot loss & accuracy score over the course of training 

    '''
    fig, ax = plt.subplots(nrows = 2, ncols = 1) 
    plt.subplot(2, 1, 1)
    plt.plot(range(epoch+1), test_losses)
    plt.title('Training of SweetNet')
    plt.ylabel('Validation Loss')
    plt.legend(['Validation Loss'],loc = 'best')

    plt.subplot(2, 1, 2)
    plt.plot(range(epoch+1), test_acc)
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Number of Epochs')
    plt.legend(['Validation Accuracy'], loc = 'best')
    plt.show()
    
    return model, performance_metric, best_model_wts, test_losses, test_acc, epoch
    '''

def init_weights(m):
    # print(m)
    if type(m) == torch.nn.Linear:
        # print('yes, type of m is nn.Linear')
        torch.nn.init.sparse_(m.weight, sparsity = 0.1)
        # print(torch.nn.init.sparse_(m.weight, sparsity = 0.1))

#data = from_smiles('CCO')  # Convert ethanol SMILES string to Data object

#print(data)
#print(list(data))

# df = pd.read_csv('/home/ssh23/Documents/thesis/Data_processing/GNN/toy_data.csv', sep=",", quotechar='"')
# data = pd.read_csv('./data/iupac_to_smiles.csv', sep=",", quotechar='"')
data = pd.read_csv('../data/nonrectified_iupac_to_smiles.csv', sep=",", quotechar='"')
print(data)
print(data.columns)

rank = 'Domain'
X, rank, splits, class_list, _ = hierarchy_filter(data, rank = 'Domain', min_seq = 5, wildcard_seed = False, wildcard_list = None,
                                                                                          wildcard_name = None, r = 0.1)
print(X)
print(splits)
# print(class_list)
# smiles_list = (data.smiles.values.tolist())

seed = 42
# torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
np.random.seed(seed)
random.seed(seed)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



best_model = None
result_list = []

for conv_type in ['GraphConv', 'gin', 'sage', 'gcn']:
    best_overall_acc = 0.0
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    print(f"training model with {conv_type} convolution")
    torch.manual_seed(42)    
    input_dim = 9  # Number of node features
    hidden_dim = 128
    # output_dim = 1  # For regression or binary classification
    # num_classes = len(class_list)  # Number of classes for classification, use 1 for regression
    # print('before entering into model', num_classes)

    for split_num, (i, j) in enumerate(splits.split(X.smiles.values.tolist(), X[rank].values.tolist())):
        print(f"split {split_num} is taking place")
        print('^-^' * (split_num+1))
        #print(i, j)
        #print(df.smiles.values.tolist())
        
        train_x = [X.smiles.values.tolist()[k] for k in i]
        train_y = [X[rank].values.tolist()[k] for k in i]
        #print(train_x)
        #print(len(train_x))
        len_train_x = [len(k) for k in train_x]
        #print((len_train_x))
        test_x = [X.smiles.values.tolist()[k] for k in j]
        test_y = [X[rank].values.tolist()[k] for k in j]
        #print(test_x)
        #print(len(test_x))
        id_val = list(range(len(test_x)))
        len_val_x = [len(k) for k in test_x]
    
        id_val = [[id_val[k]] * len_val_x[k] for k in range(len(len_val_x))]
        id_val = [item for sublist in id_val for item in sublist]
        
        train_data_list, classes = from_smiles_list(train_x, train_y)
        test_data_list, classes = from_smiles_list(test_x, test_y)
        classes = set(classes)
        print(len(classes))

        train_loader = DataLoader(train_data_list, batch_size = 32, shuffle = True, drop_last=True)
        test_loader = DataLoader(test_data_list, batch_size = 32, shuffle = False, drop_last=True)

        dataloaders = {'train' : train_loader, 'test' : test_loader}     

        num_classes = len(classes)  # Number of classes for classification, use 1 for regression
        print('before entering into model', num_classes)
        model = SuperSweetNet(input_dim, hidden_dim, num_classes, conv_type=conv_type)
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"number of parameters are: {num_params}")
        model.apply(init_weights)
        model.to(device)
        early_stopping = EarlyStopping(patience = 50, verbose = True)
        optimizer_ft = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay=0.001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, 50)
        criterion = torch.nn.CrossEntropyLoss().to(device)
    
        model_ft, performance_metric, best_model_wts, test_losses, test_acc, test_mcc, epoch = train_model(model, criterion, 
                                                                                            optimizer_ft, scheduler, 
                                                                                            dataloaders, conv_type, num_epochs=100)
        
        acc_list.append(performance_metric['accuracy'])
        precision_list.append(performance_metric['precision'])
        recall_list.append(performance_metric['recall'])
        f1_list.append(performance_metric['f1-score'])
        auc_list.append([performance_metric['auc']])
        
        if performance_metric['accuracy'] > best_overall_acc:
            best_overall_acc = performance_metric['accuracy']

            ## save model
            print('best accuracy is: ', best_overall_acc)
            model_path = Path('models')
            model_path.mkdir(parents = True, exist_ok=True)
            model_name = f'best_{conv_type}_model.pth'
            model_save_path = model_path / model_name
            model.load_state_dict(best_model_wts)          
            torch.save(best_model_wts, model_save_path)
            
            ## save plots

            fig, ax = plt.subplots(nrows = 1, ncols = 1) 
            plt.subplot(1, 1, 1)
            plt.plot(range(epoch+1), test_mcc)
            plt.title('Training of SuperSweetNet')
            plt.ylabel('Validation MCC')
            plt.xlabel('Number of Epochs')
            plt.legend(['MCC score'],loc = 'best')

            """
            fig, ax = plt.subplots(nrows = 2, ncols = 1) 
            plt.subplot(2, 1, 1)
            plt.plot(range(epoch+1), test_losses)
            plt.title('Training of SuperSweetNet')
            plt.ylabel('Validation Loss')
            plt.legend(['Validation Loss'],loc = 'best')

            plt.subplot(2, 1, 2)
            plt.plot(range(epoch+1), test_acc)
            plt.ylabel('Validation Accuracy')
            plt.xlabel('Number of Epochs')
            plt.legend(['Validation Accuracy'], loc = 'best')
            # plt.show()
            """
            
            plt.tight_layout()
            
            plot_path = Path('v1_mcc_plots')
            plot_path.mkdir(parents = True, exist_ok=True)
            output_path = f'./v1_mcc_plots/output_{rank}_{conv_type}_plot.png' 

            plt.savefig(output_path)
            print("Plot saved as 'output_plot.png'")
            
        # break
    # break


    # for accuracy        
    mean_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    accuracy_str = f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"

    # for precision
    mean_precision = np.mean(precision_list)
    std_precision = np.std(precision_list)
    precision_str = f"{mean_precision:.4f} ± {std_precision:.4f}"

    # for recall 
    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)
    recall_str = f"{mean_recall:.4f} ± {std_recall:.4f}"

    # for f1-score
    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    f1_str = f"{mean_f1:.4f} ± {std_f1:.4f}"

    # for auc
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)
    auc_str = f"{mean_auc:.4f} ± {std_auc:.4f}"



    # Print the result as mean ± std deviation
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    result_list.append({"Model": conv_type+'_SuSNet',
                        'Classification': rank,
                        'Accuracy': accuracy_str,
                        'Precision' : precision_str,
                        'Recall' : recall_str,
                        'F1-Score' : f1_str,
                        'AUC' : auc_str})        
    
print(result_list)
df_results = pd.DataFrame(result_list)
# Save the table as a CSV file
df_results.to_csv('model_accuracies.csv', index=False)

# Print the DataFrame
print(df_results)



"""
# ***************************************
def get_full_data_emebeddings(model, dataloader):
    embeddings_list = []
    for phase in ['train', 'test']:
        for data in dataloader[phase]:
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


embeddings = {}
for conv_type in ['GraphConv', 'gin', 'sage', 'gcn']:
    
    model_ft = SuperSweetNet(input_dim, hidden_dim, num_classes, conv_type=conv_type)
    print(conv_type)
    model_filename = f'models/best_{conv_type}_model.pth'
    print(model_filename)
    model_ft.load_state_dict(torch.load(model_filename))
    embeddings_df = pd.DataFrame(np.concatenate(get_full_data_emebeddings(model_ft, dataloaders)))
    print('embeddings for: ', conv_type)
    print(embeddings_df)
    embeddings[conv_type] = embeddings_df
    # print(embeddings)
    
    # break

print(embeddings)
print(embeddings.keys())
for k, v in embeddings.items():
    print(k, v)
    print(v.shape)
    tsne_emb = TSNE(random_state=42).fit_transform(v)
    print(tsne_emb)
    plt.figure(figsize=(9,9))
    sns.scatterplot(x = tsne_emb[:,0], y = tsne_emb[:,1], s = 40, alpha = 0.4,
                hue = data.Kingdom.values.tolist(), palette = 'colorblind', rasterized = True)
    print('scatter plot is on the way')
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xlabel('t-SNE Dim1')
    plt.ylabel('t-SNE Dim2')
    plt.title('Kingdom')
    plt.tight_layout()
    
    plot_path = Path('embeddings')
    plot_path.mkdir(parents = True, exist_ok=True)
    output_path = f'./embeddings/output_{rank}_{conv_type}_plot.png' 

    plt.savefig(output_path)
    print("Plot saved as 'output_plot.png'")
     
"""


"""
             Model  Classification         Accuracy
0  GraphConvSuSNet  immunogenicity  0.9421 ± 0.0157
1        ginSuSNet  immunogenicity  0.9214 ± 0.0099
2       sageSuSNet  immunogenicity  0.9396 ± 0.0148
3        gcnSuSNet  immunogenicity  0.9339 ± 0.0107

********* Results of sugarbase dataset **************

              Model Classification         Accuracy
0  GraphConv_SuSNet        Species  0.4653 ± 0.0278
1        gin_SuSNet        Species  0.4038 ± 0.0181
2       sage_SuSNet        Species  0.4769 ± 0.0181
3        gcn_SuSNet        Species  0.4386 ± 0.0096
        0  SweetNet        Species  0.6144 ± 0.0135

              Model Classification         Accuracy
0  GraphConv_SuSNet          Genus  0.4539 ± 0.0181
1        gin_SuSNet          Genus  0.4037 ± 0.0167
2       sage_SuSNet          Genus  0.4735 ± 0.0204
3        gcn_SuSNet          Genus  0.4241 ± 0.0158
0          SweetNet          Genus  0.6091 ± 0.0125

              Model Classification         Accuracy
0  GraphConv_SuSNet         Family  0.4730 ± 0.0239
1        gin_SuSNet         Family  0.4171 ± 0.0144
2       sage_SuSNet         Family  0.4735 ± 0.0231
3        gcn_SuSNet         Family  0.4377 ± 0.0109
0           SweetNet        Family  0.6141 ± 0.0091


             Model Classification         Accuracy        Precision           Recall         F1-Score              AUC
0  GraphConv_SuSNet          Class  0.5966 ± 0.0175  0.3463 ± 0.1371  0.3892 ± 0.1375  0.3522 ± 0.1366  0.0000 ± 0.0000
1        gin_SuSNet          Class  0.5712 ± 0.0134  0.2568 ± 0.1272  0.3095 ± 0.1273  0.2681 ± 0.1212  0.0000 ± 0.0000
2       sage_SuSNet          Class  0.6124 ± 0.0159  0.3462 ± 0.1095  0.3934 ± 0.1159  0.3569 ± 0.1094  0.0000 ± 0.0000
3        gcn_SuSNet          Class  0.5736 ± 0.0097  0.2991 ± 0.1372  0.3300 ± 0.1377  0.3006 ± 0.1316  0.0000 ± 0.0000
0          SweetNet          Class  0.6881 ± 0.0103

              Model Classification         Accuracy        Precision           Recall         F1-Score              AUC
0  GraphConv_SuSNet          Order  0.5019 ± 0.0264  0.2787 ± 0.0934  0.3150 ± 0.1007  0.2838 ± 0.0947  0.0000 ± 0.0000
1        gin_SuSNet          Order  0.4457 ± 0.0160  0.2095 ± 0.0914  0.2447 ± 0.0878  0.2171 ± 0.0896  0.0000 ± 0.0000
2       sage_SuSNet          Order  0.5031 ± 0.0126  0.2797 ± 0.0968  0.3150 ± 0.0881  0.2878 ± 0.0937  0.0000 ± 0.0000
3        gcn_SuSNet          Order  0.4615 ± 0.0094  0.2396 ± 0.0980  0.2777 ± 0.0961  0.2468 ± 0.0973  0.0000 ± 0.0000
0          SweetNet          Order  0.6218 ± 0.0137

              Model Classification         Accuracy        Precision           Recall         F1-Score              AUC
0  GraphConv_SuSNet         Phylum  0.6873 ± 0.0123  0.3798 ± 0.1301  0.4176 ± 0.1109  0.3885 ± 0.1233  0.0000 ± 0.0000
1        gin_SuSNet         Phylum  0.6585 ± 0.0102  0.3051 ± 0.0839  0.3457 ± 0.0849  0.3140 ± 0.0859  0.0000 ± 0.0000
2       sage_SuSNet         Phylum  0.6830 ± 0.0123  0.3366 ± 0.1226  0.3619 ± 0.1226  0.3386 ± 0.1239  0.0000 ± 0.0000
3        gcn_SuSNet         Phylum  0.6597 ± 0.0088  0.3284 ± 0.0958  0.3586 ± 0.0867  0.3316 ± 0.0905  0.0000 ± 0.0000
0          SweetNet         Phylum  0.7148 ± 0.0045


              Model Classification         Accuracy        Precision           Recall         F1-Score              AUC
0  GraphConv_SuSNet        Kingdom  0.7494 ± 0.0072  0.3528 ± 0.1318  0.4141 ± 0.1187  0.3682 ± 0.1273  0.0000 ± 0.0000
1        gin_SuSNet        Kingdom  0.7301 ± 0.0107  0.3317 ± 0.1124  0.4015 ± 0.0901  0.3560 ± 0.1038  0.0000 ± 0.0000
2       sage_SuSNet        Kingdom  0.7471 ± 0.0067  0.3305 ± 0.1031  0.4157 ± 0.1019  0.3572 ± 0.1031  0.0000 ± 0.0000
3        gcn_SuSNet        Kingdom  0.7313 ± 0.0053  0.3496 ± 0.1353  0.4250 ± 0.1178  0.3748 ± 0.1283  0.0000 ± 0.0000
0          SweetNet        Kingdom  0.7646 ± 0.0074

              Model Classification         Accuracy        Precision           Recall         F1-Score              AUC
0  GraphConv_SuSNet         Domain  0.7626 ± 0.0071  0.2499 ± 0.0603  0.3043 ± 0.0657  0.2725 ± 0.0630  0.0000 ± 0.0000
1        gin_SuSNet         Domain  0.7476 ± 0.0082  0.2482 ± 0.0585  0.3037 ± 0.0614  0.2709 ± 0.0609  0.0000 ± 0.0000
2       sage_SuSNet         Domain  0.7643 ± 0.0040  0.2510 ± 0.0543  0.3077 ± 0.0596  0.2744 ± 0.0570  0.0000 ± 0.0000
3        gcn_SuSNet         Domain  0.7498 ± 0.0048  0.2463 ± 0.0579  0.3088 ± 0.0624  0.2730 ± 0.0606  0.0000 ± 0.0000
4          SweetNet         Domain  0.7755 ± 0.0019

GraphConv_SuSNet,immunogenicity,0.9507 ± 0.0094,0.9558 ± 0.0298,0.9594 ± 0.0296,0.9567 ± 0.0299,0.0000 ± 0.0000
gin_SuSNet,immunogenicity,0.9375 ± 0.0118,0.9428 ± 0.0219,0.9386 ± 0.0277,0.9387 ± 0.0235,0.0000 ± 0.0000
sage_SuSNet,immunogenicity,0.9510 ± 0.0068,0.9611 ± 0.0321,0.9677 ± 0.0297,0.9635 ± 0.0311,0.0000 ± 0.0000
gcn_SuSNet,immunogenicity,0.9465 ± 0.0088,0.9531 ± 0.0347,0.9489 ± 0.0345,0.9492 ± 0.0346,0.0000 ± 0.0000

0  SweetNet  immunogenicity  0.9711 ± 0.0071


########### Number of parameters
SweetNet    Immuno  706114
susnet_graphconv    Immuno  ??????

SweetNet    Domain  708649
SuSNet_graphConv      Domain  402217
SuSNet_GINConv        Domain  417833
SuSNet_sage           Domain  402217
SuSNet_GCNConv        Domain  368297

SweetNet    Kingdom     709949
SuSNet_graphconv      Kingdom     403517
susnet_gin            Kingdom     419133


print((train_data_list[0]), len(test_data_list))
print(len(class_list))


print(train_data_list)
for data in train_data_list:
   print(type(data))
   print(data.num_nodes)
   print(data.is_directed())
   print(data.keys())
   print(data["x"], data['y'])
   print(data.y)
   for key, item in data:
      print('found in data', key)
   break

train_loader = DataLoader(train_data_list, batch_size = 3, shuffle = True)
test_loader = DataLoader(test_data_list, batch_size = 3, shuffle = False)
"""



# # Set the random seed for reproducibility
# def set_seed(random_seed):
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     np.random.seed(random_seed)
#     random.seed(random_seed)

# set_seed(42)  # or any seed you prefer

# train_loader = DataLoader(train_data_list, batch_size = 32, shuffle = True) #, drop_last=True)
# test_loader = DataLoader(test_data_list, batch_size = 32, shuffle = False) #, drop_last=True)
# print(train_loader)

# dataloaders = {'train' : train_loader, 'test' : test_loader}

"""
dataiter = iter(train_loader)
data = next(dataiter)
features, labels, batch = data.x, data.y, data.batch

print(features, labels, batch)
for batch in train_loader:
    print(batch)
    inputs =batch.x
    targets =batch.y
    print(inputs.shape, targets.shape)

    if inputs.size(0) != targets.size(0):
        print(f"Mismatch in batch size: Inputs batch size {inputs.size(0)}, Targets batch size {targets.size(0)}")
        raise ValueError("Mismatch")
    print(batch.shape)
    print(batch.num_graphs)
    print("Batch labels:", batch.y.tolist())
    print("Batch features:", batch.x)
    break
"""

'''
import math
# training loop
num_epoch = 2
total_samples = len(train_data_list)
n_iteration = math.ceil(total_samples/3)
# print(total_samples, n_iteration)
'''








### Once Data is generated, apply GNN on it

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, TopKPooling, SAGEConv, GINConv
# from torch_geometric.data import Data, DataLoader
# from torch.nn import Sequential as seq, ReLU, Linear as linear

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class SuperSweetNet(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, conv_type = 'GCNConv'):
#         super(SuperSweetNet, self).__init__()


#         if conv_type == 'sage':
#             self.conv1 = SAGEConv(input_dim, hidden_dim)
#             self.conv2 = SAGEConv(hidden_dim, hidden_dim)
#             self.conv3 = SAGEConv(hidden_dim, hidden_dim)

#         elif conv_type == 'gcn':
#             self.conv1 = GCNConv(input_dim, hidden_dim)
#             self.conv2 = GCNConv(hidden_dim, hidden_dim)
#             self.conv3 = GCNConv(hidden_dim, hidden_dim)

#         elif conv_type == 'gin':
#             self.conv1 = GINConv(seq(linear(input_dim, hidden_dim), ReLU(), linear(hidden_dim, hidden_dim)))
#             self.conv2 = GINConv(seq(linear(hidden_dim, hidden_dim), ReLU(), linear(hidden_dim, hidden_dim)))
#             self.conv3 = GINConv(seq(linear(hidden_dim, hidden_dim), ReLU(), linear(hidden_dim, hidden_dim)))

#         elif conv_type == 'GraphConv':
#             self.conv1 = GraphConv(input_dim, hidden_dim)
#             self.conv2 = GraphConv(hidden_dim, hidden_dim)
#             self.conv3 = GraphConv(hidden_dim, hidden_dim)


#         # self.conv1 = GCNConv(input_dim, hidden_dim)
#         '''
#         input_dim: The number of input features per node. This tells the layer how many features each node 
#         in the input graph has.
#         hidden_dim: The number of output features per node. This determines the dimensionality of the 
#         feature vectors for each node after the convolution operation.'''
#         self.pool1 = TopKPooling(hidden_dim, ratio=0.8)
#         # self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.pool2 = TopKPooling(hidden_dim, ratio=0.8)
#         # self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.pool3 = TopKPooling(hidden_dim, ratio=0.8)
#         self.lin1 = linear(hidden_dim * 2, 1024)
#         self.lin2 = linear(1024, 64)
#         self.lin3 = linear(64, num_classes)
#         self.bn1 = torch.nn.BatchNorm1d(1024)
#         self.bn2 = torch.nn.BatchNorm1d(64)
#         self.act1 = torch.nn.LeakyReLU()
#         self.act2 = torch.nn.LeakyReLU()

#     def forward(self, x, edge_index, batch, inference = False):
        
#         att = 0
#         # print(x.shape, x.ndim)
#         # # x = self.item_embedding(x) # 
#         # print(x.shape, x.ndim)
#         # # x = x.squeeze(1) 
#         # print(x.shape, x.ndim)
#         # x = x.reshape(-1, x.shape[0])
#         # print(x.shape, x.ndim)
#         # print(edge_index.max().item(), edge_index.min().item())
#         # print(x.size(0))

        
#         # x = x.squeeze(1)
#         # print(x.shape, x.ndim)

#         if edge_index.max().item() >= x.size(0) or edge_index.min().item() < 0:
#             # print('an error')
#             raise ValueError(f"Invalid edge_index values: {edge_index}")
#         """
#         try:
#             #print("Before conv1:")
#             #print("x:", x)
#             print(x.shape)
#             print(x.dtype, edge_index.dtype)
#             x = x.float()
#             # edge_index = edge_index.float()
#             print("x shape:", x.shape)
#             print("edge_index shape:", edge_index.shape)
#             print("edge_index:", edge_index)
#             x = self.conv1(x, edge_index)
#             print('Scenario 1, conv1 executed')
#             print(x.shape, x.ndim)
#             #print("After conv1, before activation:")
#             #print("x:", x)
#             x = F.leaky_relu(x)
#             print('leaky_relu is executed')
#             print(x.shape, x.ndim)
#             #print("After activation:")
#             #print("x:", x)
#         except RuntimeError as e:
#             print("Error during GCNConv operation:", e)
#             raise
        
#         print(x.shape)
#         print(x.dtype, edge_index.dtype)
#         """
#         x = x.float().to(device)
#         edge_index = edge_index.to(device)
#         # print("x shape:", x.shape, x.ndim)
#         # print("edge_index shape:", edge_index.shape)
#         # print("edge_index:", edge_index)
#         x = F.leaky_relu(self.conv1(x, edge_index))
#         '''
#         During the forward pass, you need to pass the actual data to the layer to perform the 
#         graph convolution operation. The GCNConv layer uses the weight matrices 
#         (initialized with input_dim and hidden_dim) to transform the input features x 
#         based on the connectivity information in edge_index.'''
#         # print('Scenario 1, conv1 and leaky relu are executed')
#         # print(x.shape, x.ndim)

#         x, edge_index, _, batch, _, _= self.pool1(x, edge_index, None, batch)
#         # print('pooling 1 is done')
#         # print('gmp', gmp(x, batch).shape, 'gap', gap(x, batch).shape)
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

#         x = F.leaky_relu(self.conv2(x, edge_index))
#         # print('Scenario 2, conv2 and leaky relu are executed')
#         # print(x.shape, x.ndim)
     
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
#         # print('pooling 2 is done')
#         x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

#         x = F.leaky_relu(self.conv3(x, edge_index))
#         # print('Scenario 3, conv3 and leaky relu are executed')
#         # print(x.shape, x.ndim)

#         x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
#         # print('pooling 3 is done')
#         x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

#         x = x1 + x2 + x3
#         # print('x = x1 + x2 + x3', x.shape, x.ndim)
#         # print(num_classes)
#         x = self.lin1(x)
#         # print('lin1 is executed', x.shape, x.ndim)
#         x = self.bn1(self.act1(x))
#         # print('bn1 is executed', x.shape, x.ndim)
#         x = self.lin2(x)
#         # print('lin2 is executed', x.shape, x.ndim)
#         x = self.bn2(self.act2(x))      
#         # print('bn1 is executed', x.shape, x.ndim)
#         x = F.dropout(x, p = 0.5, training = self.training)
#         # print(x.shape, x.ndim)

#         x = self.lin3(x).squeeze(1)
#         # print('lin3 is executed', x.shape, x.ndim)

#         if inference:
#           x_out = x1 + x2 + x3
#         #   print(x_out, x_out.shape, x_out.ndim)
#           return x, x_out, att
#         else:
#         #   print("printing before returning", x.shape, x.ndim)
#           return x.to(device)
        

#     def get_embeddings(self, x, edge_index, batch, inference=True):
#         """
#         extract the final embeddings before the final classification layer"""

#         x = x.float().to(device)
#         edge_index = edge_index.to(device)

#         x = F.leaky_relu(self.conv1(x, edge_index))
#         '''
#         During the forward pass, you need to pass the actual data to the layer to perform the 
#         graph convolution operation. The GCNConv layer uses the weight matrices 
#         (initialized with input_dim and hidden_dim) to transform the input features x 
#         based on the connectivity information in edge_index.'''
#         # print('Scenario 1, conv1 and leaky relu are executed')
#         # print(x.shape, x.ndim)

#         x, edge_index, _, batch, _, _= self.pool1(x, edge_index, None, batch)
#         # print('pooling 1 is done')
#         # print('gmp', gmp(x, batch).shape, 'gap', gap(x, batch).shape)
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

#         x = F.leaky_relu(self.conv2(x, edge_index))
#         # print('Scenario 2, conv2 and leaky relu are executed')
#         # print(x.shape, x.ndim)
     
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
#         # print('pooling 2 is done')
#         x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

#         x = F.leaky_relu(self.conv3(x, edge_index))
#         # print('Scenario 3, conv3 and leaky relu are executed')
#         # print(x.shape, x.ndim)

#         x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
#         # print('pooling 3 is done')
#         x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

#         x_embeddings = x1 + x2 + x3

#         return x_embeddings



# # Example usage
# #model = SuperSweetNet(input_dim=9, hidden_dim=128, output_dim=128, num_classes=8, lib_size=1000)
# #print(model)




# class EarlyStopping:
#    """Early stops the training if validation loss doesn't improve after a given patience."""
#    def __init__(self, patience = 7, verbose = False):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement. 
#                             Default: False
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = 0


#    def __call__(self, val_loss, model):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

#    def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         self.val_loss_min = val_loss


# ## Train Model
# def train_model(model, criterion, optimizer, scheduler, dataloaders, conv_type, num_epochs = 30):
#     """
#     model: trining model for instance GNN
#     criterion: 
#     optimizer:
#     scheduler,
#     num_epochs: number of times forward and backward pass
#     source: https://github.com/BojarLab/SweetNet/blob/main/SweetNet_code.ipynb"""

#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss = 100.0
#     best_acc = 0
#     test_losses = []
#     test_acc = []
#     # y_set = set()
#     # problem = 0
#     for epoch in range(num_epochs):

#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         for phase in ['train', 'test']:
#             if phase == 'train':
#                 # print('phase is train')
#                 model.train()
#             else:
#                 # print('phase is test')
#                 model.eval()

#             running_loss = []
#             running_acc = []
#             running_mcc = []
#             i = 0
#             for data in dataloaders[phase]:
#                 # print("data", data)
#                 # print(phase)
#                 # print(data)
                
#                 x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
#                 x = x.to(device)# .cuda()
#                 y = y.to(device) #.cuda()
#                 # print('x is: ', x.shape)
#                 # print('y is: ',y.shape)
#                 # i += 1
#                 # print('batch is: ',batch, i)
#                 edge_index = edge_index.to(device) #.cuda()
#                 batch = batch.to(device) #.cuda()
#                 # print(max(batch.tolist()))

#                 """ 
#                 for debugging
#                 if max(batch.tolist()) != 63:
#                     # problem += 1
#                     # print('problem', problem)
#                     # print('list of smiles: ', len(data.smiles))
#                     num_nodes = 0
#                     for index, smile in enumerate(data.smiles):
                        
#                         mol = Chem.MolFromSmiles(smile)
#                         if mol is None:
#                             print('mol is none', index, smile)
                            
#                         else:
#                             num_nodes += mol.GetNumAtoms()
#                     print(num_nodes)
#                     break
#                 """    
#                 optimizer.zero_grad() # Clears old gradients before computing new ones

#                 with torch.set_grad_enabled(phase == 'train'): # Enables gradient computation only during training
#                     # print(x.shape)
#                     pred = model(x, edge_index, batch) # Performs a forward pass to get predictions
#                     loss = criterion(pred, y)

#                     if phase == 'train':
#                         # If in training mode, performs backpropagation and updates model weights
#                         loss.backward()
#                         optimizer.step()

#                 running_loss.append(loss.item()) # Adds the loss for the current batch to the list
#                 pred2 = np.argmax(pred.cpu().detach().numpy(), axis = 1) # Gets the predicted class labels by taking the argmax of predictions.
#                 running_acc.append(accuracy_score( # computes and stores acc
#                                    y.cpu().detach().numpy().astype(int), pred2)) 
#                 running_mcc.append(matthews_corrcoef(y.detach().cpu().numpy(), pred2)) # computes and stores matthews corr. coef.
#                 # break # remove this for full training
#             epoch_loss = np.mean(running_loss) # Averages the loss over all batches for the epoch.
#             epoch_acc = np.mean(running_acc) # Averages the acc over all batches for the epoch.
#             epoch_mcc = np.mean(running_mcc) # Averages the mcc over all batches for the epoch.
#             print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(
#                     phase, epoch_loss, epoch_acc, epoch_mcc))
#             if phase == 'test' and epoch_loss < best_loss:
#                 best_loss = epoch_loss
#                 best_model_wts = copy.deepcopy(model.state_dict())

#             if phase == 'test' and epoch_acc > best_acc:
#                 best_acc = epoch_acc

#             if phase == 'test':
#                 test_losses.append(epoch_loss)
#                 test_acc.append(epoch_acc)
#                 early_stopping(epoch_loss, model)

#             scheduler.step()
#         # break # remove this for full training
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#         # print()
        
    

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#       time_elapsed // 60, time_elapsed % 60))
#     print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))

#     # create folder for saving models
#     model_path = Path('models')
#     model_path.mkdir(parents = True, exist_ok=True)

#     #create model save path
#     model_name = f'best_{conv_type}_model.pth'
#     model_save_path = model_path / model_name
#     model.load_state_dict(best_model_wts)

#     torch.save(best_model_wts, model_save_path)

#     ## plot loss & accuracy score over the course of training 
#     fig, ax = plt.subplots(nrows = 2, ncols = 1) 
#     plt.subplot(2, 1, 1)
#     plt.plot(range(epoch+1), test_losses)
#     plt.title('Training of SweetNet')
#     plt.ylabel('Validation Loss')
#     plt.legend(['Validation Loss'],loc = 'best')

#     plt.subplot(2, 1, 2)
#     plt.plot(range(epoch+1), test_acc)
#     plt.ylabel('Validation Accuracy')
#     plt.xlabel('Number of Epochs')
#     plt.legend(['Validation Accuracy'], loc = 'best')
#     return model, best_acc, best_loss

    # model_path = Path('models')
    # model_path.mkdir(parents=True, exist_ok = True)
    
    # # 2. create model sav epath
    # model_name = 'best_model.pth'
    # model_save_path = model_path / model_name
    # #print(model_save_path)

    # # 3. save the model.state_dict()
    # torch.save(obj = model.state_dict(), f = model_save_path)





## Model Training
# input_dim = 9  # Number of node features
# hidden_dim = 128
# # output_dim = 1  # For regression or binary classification
# num_classes = len(class_list)  # Number of classes for classification, use 1 for regression
# print('before entering into model', num_classes)

"""
lib_size = len(class_list)  # Example library size
model = SuperSweetNet(input_dim, hidden_dim, output_dim, num_classes, conv_type='sage')
print(model)

model.to(device) #.cuda()
early_stopping = EarlyStopping(patience = 50, verbose = True)
optimizer_ft = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay=0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, 50)
criterion = torch.nn.CrossEntropyLoss().to(device) #.cuda()
for batch in train_loader:
    x, edge_index, batch = batch.x, batch.edge_index, batch.batch
    assert (edge_index.max() < x.size(0))
    
model_ft = train_model(model, criterion, optimizer_ft, scheduler, dataloaders, 
                   num_epochs = 1)
"""

# # torch.manual_seed(42)
# seed = 42
# # torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
# np.random.seed(seed)
# random.seed(seed)

# # Ensure deterministic behavior
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# def init_weights(m):
#     # print(m)
#     if type(m) == torch.nn.Linear:
#         # print('yes, type of m is nn.Linear')
#         torch.nn.init.sparse_(m.weight, sparsity = 0.1)
#         # print(torch.nn.init.sparse_(m.weight, sparsity = 0.1))


# for conv_type in ['GraphConv', 'gin', 'sage', 'gcn']:
#     print(f"training model with {conv_type} convolution")
#     torch.manual_seed(42)

#     model = SuperSweetNet(input_dim, hidden_dim, num_classes, conv_type=conv_type)
#     print(model)
#     num_params = sum(p.numel() for p in model.parameters())
#     print(f"number of parameters are: {num_params}")
    
    
#     model.apply(init_weights)
#     model.to(device)
#     early_stopping = EarlyStopping(patience = 50, verbose = True)
#     optimizer_ft = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay=0.001)

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, 50)
#     criterion = torch.nn.CrossEntropyLoss().to(device)
    
#     model_ft = train_model(model, criterion, optimizer_ft, scheduler, dataloaders,
#                            conv_type, num_epochs=100)
    
    
    
'''
# ***************************************
def get_full_data_emebeddings(model, dataloader):
    embeddings_list = []
    for phase in ['train', 'test']:
        for data in dataloader[phase]:
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


embeddings = {}
for conv_type in ['GraphConv', 'gin', 'sage', 'gcn']:
    
    model_ft = SuperSweetNet(input_dim, hidden_dim, num_classes, conv_type=conv_type)
    print(conv_type)
    model_filename = f'models/best_{conv_type}_model.pth'
    print(model_filename)
    model_ft.load_state_dict(torch.load(model_filename))
    embeddings_df = pd.DataFrame(np.concatenate(get_full_data_emebeddings(model_ft, dataloaders)))
    print('embeddings for: ', conv_type)
    print(embeddings_df)
    embeddings[conv_type] = embeddings_df
    # print(embeddings)

print(embeddings)
print(embeddings.keys())
'''

"""
model_ft = SuperSweetNet(input_dim, hidden_dim, output_dim, num_classes, conv_type = 'sage') # initialize model
model_ft.load_state_dict(torch.load('models/best_model.pth')) # load saved model into initialized model

model_ft.eval()

embeddings_df = pd.DataFrame(np.concatenate(get_full_data_emebeddings(model_ft, dataloaders)))
print('final embeddings are gives as: ', embeddings_df)



def indexed_classes(df, rank: str):
    # print(type(df.rank))
    total_classes = len(df[rank].values.tolist())
    print(total_classes)
    abc = list(range(total_classes))
    return abc

classes = indexed_classes(df, 'Species')
print(len(classes))


glycans_mol = from_smiles_list(df, classes)
glycan_loader = DataLoader(glycans_mol, batch_size = 64, shuffle = False, drop_last=True)



embeddings_list = []



# Set the model to evaluation mode
model.eval()

# Forward pass to get embeddings
x = x.to(device)
edge_index = edge_index.to(device)
batch = batch.to(device)
with torch.no_grad():
    output, embeddings, attribute = model(x, edge_index, batch, inference=True)

print("Embeddings:", embeddings)
print("Shape of embeddings:", embeddings.shape)
"""


























