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
data = pd.read_csv('../data/iupac_to_smiles.csv', sep=",", quotechar='"')
#data = pd.read_csv('../data/nonrectified_iupac_to_smiles.csv', sep=",", quotechar='"')
print(data)
print(data.columns)

rank = 'Domain'
X, rank, splits, class_list = hierarchy_filter(data, rank = 'Domain', min_seq = 5, wildcard_seed = False, wildcard_list = None,
                                                                                          wildcard_name = None, r = 0.1)
print(X)
print(splits)
#print(class_list)
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


'''
** Parameters **
SuSNet_GraphConv    Domain  399812  28190(data points left)     4/4: # of classes
SuSNet_Gin    Domain  415428   Accuracy: 0.9226 ± 0.0092
SuSNet_sage   Domain    399812 Accuracy: 0.9411 ± 0.0010
SuSNet_GCNConv  Domain  365892
SweetNet        Domain  706244

[{'Model': 'GraphConv_SuSNet', 'Classification': 'Domain', 'Accuracy': '0.9309 ± 0.0010'}, 
{'Model': 'gin_SuSNet', 'Classification': 'Domain', 'Accuracy': '0.9226 ± 0.0092'}, 
{'Model': 'sage_SuSNet', 'Classification': 'Domain', 'Accuracy': '0.9411 ± 0.0010'}, 
{'Model': 'gcn_SuSNet', 'Classification': 'Domain', 'Accuracy': '0.9258 ± 0.0026'}]


SuSNet_GraphConv    Kingdom    400527      28183      15/19  
SuSNet_Gin    Kingdom  416143      Accuracy: 0.8640 ± 0.0022
SuSNet_sage   Kingdom    400527    Accuracy: 0.8815 ± 0.0018
SuSNet_GCNConv  Kingdom  366607
SweetNet        Kingdom  706959

[{'Model': 'GraphConv_SuSNet', 'Classification': 'Kingdom', 'Accuracy': '0.8777 ± 0.0036'}, 
{'Model': 'gin_SuSNet', 'Classification': 'Kingdom', 'Accuracy': '0.8640 ± 0.0022'}, 
{'Model': 'sage_SuSNet', 'Classification': 'Kingdom', 'Accuracy': '0.8815 ± 0.0018'}, 
{'Model': 'gcn_SuSNet', 'Classification': 'Kingdom', 'Accuracy': '0.8663 ± 0.0072'}]



SuSNet_GraphConv    Phylum    402932      28166      52/63  
SuSNet_Gin    Phylum  418548      0.7571 ± 0.0062 (ACC.)
SuSNet_sage   Phylum    402932    
SuSNet_GCNConv  Phylum  369012
SweetNet        Phylum  709299
[{'Model': 'GraphConv_SuSNet', 'Classification': 'Phylum', 'Accuracy': '0.7786 ± 0.0202'}, 
{'Model': 'gin_SuSNet', 'Classification': 'Phylum', 'Accuracy': '0.7571 ± 0.0062'}, 
{'Model': 'sage_SuSNet', 'Classification': 'Phylum', 'Accuracy': '0.7932 ± 0.0045'}, 
{'Model': 'gcn_SuSNet', 'Classification': 'Phylum', 'Accuracy': '0.7719 ± 0.0091'}]


SuSNet_GraphConv    Class    406377      28110      105/146  Accuracy: 0.6501 ± 0.0022
SuSNet_Gin    Class  421993      Accuracy: 0.6335 ± 0.0058
SuSNet_sage   Class    406377    Accuracy: 0.6547 ± 0.0073
SuSNet_GCNConv  Class  372457
SweetNet        Class  712744      0.6863 ± 0.0150
{'Model': 'GraphConv_SuSNet', 'Classification': 'Class', 'Accuracy': '0.6501 ± 0.0022'},
{'Model': 'gin_SuSNet', 'Classification': 'Class', 'Accuracy': '0.6335 ± 0.0058'},
{'Model': 'sage_SuSNet', 'Classification': 'Class', 'Accuracy': '0.6547 ± 0.0073'}
{'Model': 'gcn_SuSNet', 'Classification': 'Class', 'Accuracy': '0.6352 ± 0.0108'}




SuSNet_GraphConv    Order    414437      27331      229/349  
SuSNet_Gin    Order  430053      Accuracy: 0.3867 ± 0.0047
SuSNet_sage   Order    414437    Accuracy: 0.4247 ± 0.0008
SuSNet_GCNConv  Order  380517
SweetNet        Order  720544       0.5033 ± 0.0192
{'Model': 'GraphConv_SuSNet', 'Classification': 'Order', 'Accuracy': '0.4247 ± 0.0193'}
{'Model': 'gin_SuSNet', 'Classification': 'Order', 'Accuracy': '0.3867 ± 0.0047'}, 
{'Model': 'sage_SuSNet', 'Classification': 'Order', 'Accuracy': '0.4247 ± 0.0008'}, 
{'Model': 'gcn_SuSNet', 'Classification': 'Order', 'Accuracy': '0.3968 ± 0.0005'}


SuSNet_GraphConv    Family    424447 (with dup 425032)      22679      383/681  0.3821 ± 0.0271
SuSNet_Gin    Family  440063 (440648)      Accuracy: 0.3867 ± 0.0047
SuSNet_sage   Family    424447 (425032)    Accuracy: 0.3405 ± 0.0084
SuSNet_GCNConv  Family  390527
SweetNet        Family  730944  0.4387 ± 0.0059
{'Model': 'GraphConv_SuSNet', 'Classification': 'Family', 'Accuracy': '0.3448 ± 0.0119'}
{'Model': 'gin_SuSNet', 'Classification': 'Family', 'Accuracy': '0.3114 ± 0.0086'}
{'Model': 'sage_SuSNet', 'Classification': 'Family', 'Accuracy': '0.3583 ± 0.0101'}
{'Model': 'gcn_SuSNet', 'Classification': 'Family', 'Accuracy': '0.3227 ± 0.0051'}



With duplicates: SuSNet_GraphConv    Genus    441282      26197      642/1352  0.3166 ± 0.0107
Without duplicates: SuSNet_GraphConv    Genus    439852      23967      620/1352  0.3166 ± 0.0107
SuSNet_Gin    Genus  455468 (456898)      Accuracy: 0.2750 ± 0.0064
SuSNet_sage   Genus    439852 (441282)    Accuracy: 777
SuSNet_GCNConv  Genus  405932
SweetNet        Genus  746739       0.3827 ± 0.0027
{'Model': 'GraphConv_SuSNet', 'Classification': 'Genus', 'Accuracy': '0.2977 ± 0.0130'}
{'Model': 'gin_SuSNet', 'Classification': 'Genus', 'Accuracy': '0.2699 ± 0.0048'}
{'Model': 'sage_SuSNet', 'Classification': 'Genus', 'Accuracy': '0.2931 ± 0.0215'}
{'Model': 'gcn_SuSNet', 'Classification': 'Genus', 'Accuracy': '0.2773 ± 0.0072'}


SuSNet_GraphConv    Species    456167 (wiht duplicates 441282)      24311      871/2480   0.2862 ± 0.0028
SuSNet_Gin    Species  471783 (471848)      Accuracy: 777
SuSNet_sage   Species    456167    Accuracy: 777
SuSNet_GCNConv  Species  422247
SweetNet        Species  763509       0.3631 ± 0.0131
{'Model': 'GraphConv_SuSNet', 'Classification': 'Species', 'Accuracy': '0.2804 ± 0.0146'}, 
{'Model': 'gin_SuSNet', 'Classification': 'Species', 'Accuracy': '0.2503 ± 0.0077'}, 
{'Model': 'sage_SuSNet', 'Classification': 'Species', 'Accuracy': '0.2879 ± 0.0124'}, 
{'Model': 'gcn_SuSNet', 'Classification': 'Species', 'Accuracy': '0.2598 ± 0.0096'}


'''