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

# Ignore all warnings
warnings.filterwarnings("ignore")

"""
x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

def from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch_geometric.data.Data':
    from rdkit import Chem, RDLogger
    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def from_smiles_list(smiles_list: List[str], with_hydrogen: bool = False, kekulize: bool = False) -> List[Data]:
    data_list = []
    for smiles in smiles_list:
        data = from_smiles(smiles, with_hydrogen, kekulize)
        data_list.append(data)
    return data_list
"""

#data = from_smiles('CCO')  # Convert ethanol SMILES string to Data object

#print(data)
#print(list(data))

# df = pd.read_csv('/home/ssh23/Documents/thesis/Data_processing/GNN/toy_data.csv', sep=",", quotechar='"')
data = pd.read_csv('./data/iupac_to_smiles.csv', sep=",", quotechar='"')
print(data)
#print(df.columns)

# data = preprocess_data(data, 'Species')
# print(data)


train_x,  train_y, test_x,test_y, id_val, class_list, class_converter = hierarchy_filter(data, rank = 'Species', min_seq = 5, wildcard_seed = False, wildcard_list = None,
                                                                                          wildcard_name = None, r = 0.1)

# print(class_list)
smiles_list = (data.smiles.values.tolist())

train_data_list = from_smiles_list(train_x, train_y)
test_data_list = from_smiles_list(test_x, test_y)

"""
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

train_loader = DataLoader(train_data_list, batch_size = 32, shuffle = True) #, drop_last=True)
test_loader = DataLoader(test_data_list, batch_size = 32, shuffle = False) #, drop_last=True)
# print(train_loader)

dataloaders = {'train' : train_loader, 'test' : test_loader}

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

import math
# training loop
num_epoch = 2
total_samples = len(train_data_list)
n_iteration = math.ceil(total_samples/3)
# print(total_samples, n_iteration)









### Once Data is generated, apply GNN on it

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
        input_dim: The number of input features per node. This tells the layer how many features 
        each node in the input graph has.
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
            i = 0
            for data in dataloaders[phase]:
                # print("data", data)
                # print(phase)
                # print(data)
                
                x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
                x = x.to(device)# .cuda()
                y = y.to(device) #.cuda()
                print('x is: ', x.shape)
                # print('y is: ',y.shape)
                # i += 1
                # print('batch is: ',batch, i)
                edge_index = edge_index.to(device) #.cuda()
                batch = batch.to(device) #.cuda()
                # print(max(batch.tolist()))

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
                # break # remove this for full training
            epoch_loss = np.mean(running_loss) # Averages the loss over all batches for the epoch.
            epoch_acc = np.mean(running_acc) # Averages the acc over all batches for the epoch.
            epoch_mcc = np.mean(running_mcc) # Averages the mcc over all batches for the epoch.
            print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_mcc))
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc

            if phase == 'test':
                test_losses.append(epoch_loss)
                test_acc.append(epoch_acc)
                early_stopping(epoch_loss, model)

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

    # create folder for saving models
    model_path = Path('models')
    model_path.mkdir(parents = True, exist_ok=True)

    #create model save path
    model_name = f'best_{conv_type}_model.pth'
    model_save_path = model_path / model_name
    model.load_state_dict(best_model_wts)

    torch.save(best_model_wts, model_save_path)

    ## plot loss & accuracy score over the course of training 
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
    return model

    # model_path = Path('models')
    # model_path.mkdir(parents=True, exist_ok = True)
    
    # # 2. create model sav epath
    # model_name = 'best_model.pth'
    # model_save_path = model_path / model_name
    # #print(model_save_path)

    # # 3. save the model.state_dict()
    # torch.save(obj = model.state_dict(), f = model_save_path)





## Model Training
input_dim = 9  # Number of node features
hidden_dim = 128
# output_dim = 1  # For regression or binary classification
num_classes = len(class_list)  # Number of classes for classification, use 1 for regression
print('before entering into model', num_classes)

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

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.sparse_(m.weight, sparsity = 0.1)


for conv_type in ['GraphConv', 'gin', 'sage', 'gcn']:
    print(f"training model with {conv_type} convolution")

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
    
    model_ft = train_model(model, criterion, optimizer_ft, scheduler, dataloaders,
                           conv_type, num_epochs=100)
    
    
    
    
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


























