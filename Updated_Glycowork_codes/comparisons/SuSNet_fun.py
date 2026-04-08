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
  sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state=42)
  sss.get_n_splits(df.smiles.values.tolist(), df[rank].values.tolist())
#   for split_num, (i, j) in enumerate(sss.split(df.smiles.values.tolist(), df[rank].values.tolist())):
#       print('i is: ', i)
#       print('j is: ', j)

  return df, rank, sss, class_list



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





