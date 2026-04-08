## this python file contains all the codes that are used in GNN model
## for SuperSweetNet model

from typing import Any, Dict, List

import torch

import torch_geometric
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import time, copy
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error
import matplotlib.pyplot as plt
from rdkit import Chem


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

def from_smiles(smiles: str, label: int, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch_geometric.data.Data':
    from rdkit import Chem, RDLogger
    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None: # because Rdkit could not parse it into molecule or invalid string
        mol = Chem.MolFromSmiles('')
        return 0
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
    y = torch.tensor(label, dtype=torch.long)
    
    # print(x.shape)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]] # edge_indeices.shape = (2, number of edges)
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)


def from_smiles_list(smiles_list: List[str], labels: List[int], with_hydrogen: bool = False, kekulize: bool = False) -> List[Data]:
    data_list = []
    label = []
    for index, smiles in enumerate(smiles_list):
        data = from_smiles(smiles, labels[index], with_hydrogen, kekulize)
        if data != 0:
            data_list.append((data))
            label.append(labels[index])
    return data_list, label

# **************************************
def preprocess_data(data, rank: str):
    '''
    this function removes those glycan for which rank values are not given
    data: input dataframe
    rank: taxonomic level'''
    print(data.shape)
    data = data[(data[rank] != '[]') & (data[rank] != '') & (data[rank] != ' ')]
    print(data.shape)
    return data

# **************************************



def hierarchy_filter(df_in, rank = 'Domain', min_seq = 5, wildcard_seed = False, wildcard_list = None,
                     wildcard_name = None, r = 0.1):
  """stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes
  df_in -- dataframe of glycan sequences and taxonomic labels
  rank -- which rank should be filtered; default is 'domain'
  min_seq -- how many glycans need to be present in class to keep it; default is 5
  wildcard_seed -- set to True if you want to seed wildcard glycoletters; default is False
  wildcard_list -- list which glycoletters a wildcard encompasses
  wildcard_name -- how the wildcard should be named in the IUPACcondensed nomenclature
  r -- rate of replacement, default is 0.1 or 10%
  source: https://github.com/BojarLab/SweetNet/blob/main/glycowork.py"""
  df = copy.deepcopy(df_in)
  #print(df)
  #print(rank)
  rank_list = list(df_in.columns)
  col_to_remove = {'smiles', 'glycan', rank}
  rank_list = list(set(rank_list) - col_to_remove)
  #rank_list.remove(rank)
  #rank_list.remove('smiles')
  df.drop(rank_list, axis = 1, inplace = True)
  print(df)
  class_list = sorted(set(df[rank].values.tolist()), key = str)
#   print(class_list[0])
  class_list = [item for item in class_list if item not in ('[]', '', ' ')]
#   print('length of class list is: ', len(class_list))
  temp = []

  for i in range(len(class_list)):
    t = df[df[rank] == class_list[i]]
    # print(class_list[i])
    # print(t)
    # break
    #print(type(t))
    t = t.drop_duplicates('glycan', keep = 'first')
    temp.append(t)
#   print('length of temp is: ', len(temp))
  df = pd.concat(temp).reset_index(drop = True)
#   print(df.shape)
  counts = df[rank].value_counts()
  #print(counts)
  #print(counts.index.tolist())
  #print((counts).values.tolist())
  """it removes all those glycans that are belonging to certain class with frequencz less than 5
  """
  allowed_classes = [counts.index.tolist()[k] for k in range(len(counts.index.tolist())) if (counts >= min_seq).values.tolist()[k]]
  df = df[df[rank].isin(allowed_classes)]
  print(df.shape)

#   class_list = list(sorted(class_list))
  class_list = list(sorted(list(set(df[rank].values.tolist()))))
  print(len(class_list))

  class_converter = {class_list[k]:k for k in range(len(class_list))}
  df[rank] = [class_converter[k] for k in df[rank].values.tolist()]
  classes = list((df[rank].values.tolist()))
#   print('class list is:', classes)
#   print(max(classes), min(classes))
  #print(df.shape)
  #print(df[''])
  df = df.reset_index(drop=True)
  sss = StratifiedShuffleSplit(n_splits = 3, test_size = 0.3, random_state=42)
  sss.get_n_splits(df.smiles.values.tolist(), df[rank].values.tolist())
#   for split_num, (i, j) in enumerate(sss.split(df.smiles.values.tolist(), df[rank].values.tolist())):
#       print('i is: ', i)
#       print('j is: ', j)

  return df, rank, sss, class_list, class_converter


  #print(df)
  for i, j in sss.split(df.smiles.values.tolist(), df[rank].values.tolist()):
      #print(i, j)
      #print(df.smiles.values.tolist())
      train_x = [df.smiles.values.tolist()[k] for k in i]
      train_y = [df[rank].values.tolist()[k] for k in i]
      #print(train_x)
      #print(len(train_x))
      len_train_x = [len(k) for k in train_x]
      #print((len_train_x))
      test_x = [df.smiles.values.tolist()[k] for k in j]
      test_y = [df[rank].values.tolist()[k] for k in j]
      #print(test_x)
      #print(len(test_x))
      id_val = list(range(len(test_x)))
      len_val_x = [len(k) for k in test_x]
 
      id_val = [[id_val[k]] * len_val_x[k] for k in range(len(len_val_x))]
      id_val = [item for sublist in id_val for item in sublist]

      print(len(train_x), len(train_y), len(test_x), len(test_y))

  
  return train_x, train_y, test_x, test_y, id_val, class_list, class_converter


# ********************

def Identities(df_in, rank = 'Domain'):
  """stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes
  df_in -- dataframe of glycan sequences and taxonomic labels
  rank -- which rank should be filtered; default is 'domain'
  min_seq -- how many glycans need to be present in class to keep it; default is 5
  wildcard_seed -- set to True if you want to seed wildcard glycoletters; default is False
  wildcard_list -- list which glycoletters a wildcard encompasses
  wildcard_name -- how the wildcard should be named in the IUPACcondensed nomenclature
  r -- rate of replacement, default is 0.1 or 10%
  source: https://github.com/BojarLab/SweetNet/blob/main/glycowork.py"""
  df = copy.deepcopy(df_in)
  #print(df)
  #print(rank)
  rank_list = list(df_in.columns)
  col_to_remove = {'smiles', 'glycan', rank}
  rank_list = list(set(rank_list) - col_to_remove)
  #rank_list.remove(rank)
  #rank_list.remove('smiles')
  df.drop(rank_list, axis = 1, inplace = True)
  print(df)
  class_list = sorted(set(df[rank].values.tolist()), key = str)
#   print(class_list[0])
  class_list = [item for item in class_list if item not in ('[]', '', ' ')]
#   print('length of class list is: ', len(class_list))
  temp = []

  for i in range(len(class_list)):
    t = df[df[rank] == class_list[i]]
    # print(class_list[i])
    # print(t)
    # break
    #print(type(t))
    t = t.drop_duplicates('glycan', keep = 'first')
    temp.append(t)
#   print('length of temp is: ', len(temp))
  df = pd.concat(temp).reset_index(drop = True)
#   print(df.shape)
  counts = df[rank].value_counts()
  #print(counts)
  #print(counts.index.tolist())
  #print((counts).values.tolist())
  class_list = list(sorted(list(set(df[rank].values.tolist()))))
  print(len(class_list))

  class_converter = {class_list[k]:k for k in range(len(class_list))}
  df[rank] = [class_converter[k] for k in df[rank].values.tolist()]
  classes = list((df[rank].values.tolist()))
#   print('class list is:', classes)
#   print(max(classes), min(classes))
  #print(df.shape)
  #print(df[''])
  df = df.reset_index(drop=True)
  sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3, random_state=42)
  sss.get_n_splits(df.smiles.values.tolist(), df[rank].values.tolist())
#   for split_num, (i, j) in enumerate(sss.split(df.smiles.values.tolist(), df[rank].values.tolist())):
#       print('i is: ', i)
#       print('j is: ', j)

  return df, class_list

# ***************************************


'''
def hierarchy_filter(df_in, rank = 'Domain', min_seq = 5, wildcard_seed = False, wildcard_list = None,
                     wildcard_name = None, r = 0.1):
  """stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes
  df_in -- dataframe of glycan sequences and taxonomic labels
  rank -- which rank should be filtered; default is 'domain'
  min_seq -- how many glycans need to be present in class to keep it; default is 5
  wildcard_seed -- set to True if you want to seed wildcard glycoletters; default is False
  wildcard_list -- list which glycoletters a wildcard encompasses
  wildcard_name -- how the wildcard should be named in the IUPACcondensed nomenclature
  r -- rate of replacement, default is 0.1 or 10%
  source: https://github.com/BojarLab/SweetNet/blob/main/glycowork.py"""
  df = copy.deepcopy(df_in)
  #print(df)
  #print(rank)
  rank_list = list(df_in.columns)
  col_to_remove = {'smiles', 'glycan', rank}
  rank_list = list(set(rank_list) - col_to_remove)
  #rank_list.remove(rank)
  #rank_list.remove('smiles')
  df.drop(rank_list, axis = 1, inplace = True)
#   print(df)
  class_list = sorted(set(df[rank].values.tolist()), key = str)
#   print(class_list[0])
  class_list = [item for item in class_list if item not in ('[]', '', ' ')]
#   print('length of class list is: ', len(class_list))
  temp = []

  for i in range(len(class_list)):
    t = df[df[rank] == class_list[i]]
    # print(class_list[i])
    # print(t)
    # break
    #print(type(t))
    t = t.drop_duplicates('glycan', keep = 'first')
    temp.append(t)
#   print('length of temp is: ', len(temp))
  df = pd.concat(temp).reset_index(drop = True)
#   print(df.shape)
  counts = df[rank].value_counts()
  #print(counts)
  #print(counts.index.tolist())
  #print((counts).values.tolist())
  """it removes all those glycans that are belonging to certain class with frequencz less than 5
  """
  allowed_classes = [counts.index.tolist()[k] for k in range(len(counts.index.tolist())) if (counts >= min_seq).values.tolist()[k]]
  df = df[df[rank].isin(allowed_classes)]
#   print(df.shape)

#   class_list = list(sorted(class_list))
  class_list = list(sorted(list(set(df[rank].values.tolist()))))
#   print(len(class_list))

  class_converter = {class_list[k]:k for k in range(len(class_list))}
  df[rank] = [class_converter[k] for k in df[rank].values.tolist()]
  classes = list((df[rank].values.tolist()))
#   print('class list is:', classes)
#   print(max(classes), min(classes))
  #print(df.shape)
  #print(df[''])
  df = df.reset_index(drop=True)
  sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3)
  sss.get_n_splits(df.smiles.values.tolist(), df[rank].values.tolist())
  #print(df)
  for i, j in sss.split(df.smiles.values.tolist(), df[rank].values.tolist()):
      #print(i, j)
      #print(df.smiles.values.tolist())
      train_x = [df.smiles.values.tolist()[k] for k in i]
      train_y = [df[rank].values.tolist()[k] for k in i]
      #print(train_x)
      #print(len(train_x))
      len_train_x = [len(k) for k in train_x]
      #print((len_train_x))
      test_x = [df.smiles.values.tolist()[k] for k in j]
      test_y = [df[rank].values.tolist()[k] for k in j]
      #print(test_x)
      #print(len(test_x))
      id_val = list(range(len(test_x)))
      len_val_x = [len(k) for k in test_x]
 
      id_val = [[id_val[k]] * len_val_x[k] for k in range(len(len_val_x))]
      id_val = [item for sublist in id_val for item in sublist]

      print(len(train_x), len(train_y), len(test_x), len(test_y))

  
  return train_x, train_y, test_x, test_y, id_val, class_list, class_converter
'''

# **************************************

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
# def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs = 30):
#     """
#     model: trining model for instance GNN
#     criterion: 
#     optimizer:
#     scheduler,
#     num_epochs: number of times forward and backward pass
#     source of idea: https://github.com/BojarLab/SweetNet/blob/main/SweetNet_code.ipynb"""

#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss = 100.0
#     best_acc = 0
#     test_losses = []
#     test_acc = []

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

#             for data in dataloaders[phase]:
#                 # print(phase)
#                 # print(data)
#                 x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
#                 x = x.cuda()
#                 y = y.cuda()
#                 edge_index = edge_index.cuda()
#                 batch = batch.cuda()
#                 # print('x is: ', x)
#                 # print('y is: ',y)
#                 # print('edge index is: ',edge_index)
#                 # print('batch is: ',batch)
                
                
#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     pred = model(x, edge_index, batch)
#                     loss = criterion(pred, y)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss.append(loss.item())
#                 pred2 = np.argmax(pred.cpu().detach().numpy(), axis = 1)
#                 running_acc.append(accuracy_score(
#                                    y.cpu().detach().numpy().astype(int), pred2))
#                 running_mcc.append(matthews_corrcoef(y.detach().cpu().numpy(), pred2))

#             epoch_loss = np.mean(running_loss)
#             epoch_acc = np.mean(running_acc)
#             epoch_mcc = np.mean(running_mcc)
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

#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#     print()
            
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#       time_elapsed // 60, time_elapsed % 60))
#     print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))
#     model.load_state_dict(best_model_wts)

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
    # return model
    
