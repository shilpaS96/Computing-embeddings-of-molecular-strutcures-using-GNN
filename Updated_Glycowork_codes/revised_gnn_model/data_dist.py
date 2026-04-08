import pandas as pd
import copy
import ast
from rdkit import Chem

def data_distribution(df_in, rank = 'domain', min_seq = 5):
  """stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes
  df_in -- dataframe of glycan sequences and taxonomic labels
  rank -- which rank should be filtered; default is 'domain'
  min_seq -- how many glycans need to be present in class to keep it; default is 5
  wildcard_seed -- set to True if you want to seed wildcard glycoletters; default is False
  wildcard_list -- list which glycoletters a wildcard encompasses
  wildcard_name -- how the wildcard should be named in the IUPACcondensed nomenclature
  r -- rate of replacement, default is 0.1 or 10%"""
  output_list = []
  df = copy.deepcopy(df_in)
  rank_list = list(df_in.columns)
  col_to_remove = {'glycan', rank}
  rank_list = list(set(rank_list) - col_to_remove)

  df.drop(rank_list, axis = 1, inplace = True)
  df[rank] = df[rank].apply(ast.literal_eval)
  #print(type(df[rank][148]))
  df = df.explode(rank).reset_index(drop=True)
  print('df2 from function: ', df)
  df = df.dropna(subset=[rank]).reset_index(drop=True)
  print(df)
  class_list = sorted(set(df[rank].values.tolist()), key = str)
  temp = []
  df = df.drop_duplicates(subset=['glycan', rank]).reset_index(drop=True)
  print(df.shape)
  # print(df.columns)

  #for i in range(len(class_list)):
  #  t = df[df[rank] == class_list[i]]
    # print(class_list[i])
    # print(t)
    # break
  #  t = t.drop_duplicates('target', keep = 'first')
  #  temp.append(t)
  # print('length of temp is: ', len(temp))
  #df = pd.concat(temp).reset_index(drop = True)
  # print(df.shape)
  Total_class_list = list(sorted(list(set(df[rank].values.tolist()))))

  counts = df[rank].value_counts()
  allowed_classes = [counts.index.tolist()[k] for k in range(len(counts.index.tolist())) if (counts >= min_seq).values.tolist()[k]]
  df = df[df[rank].isin(allowed_classes)]
  print(df.shape)

  class_list = list(sorted(list(set(df[rank].values.tolist()))))
  print(len(class_list))

  class_converter = {class_list[k]:k for k in range(len(class_list))}
  df[rank] = [class_converter[k] for k in df[rank].values.tolist()]
  classes = list((df[rank].values.tolist()))
  # print('class list is:', classes)
  # print(max(classes), min(classes))

  #sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state=42)
  #sss.get_n_splits(df.target.values.tolist(), df[rank].values.tolist())

  output_list.append(rank)
  output_list.append(len(class_list))
  output_list.append(df.shape[0])
  output_list.append(len(Total_class_list))
  
  return output_list
  # print(df)


data = pd.read_csv('../data/nonrectified_iupac_to_smiles.csv', sep=",", quotechar='"')
print(data.shape[0])
print(data.columns)

data = data[data['smiles'].apply(lambda smiles: Chem.MolFromSmiles(smiles) is not None)]
print(data.shape)

#data.rename(columns={'glycan': 'target'}, inplace=True)
#col_list = list(data.columns)
#for col in col_list:

Ranks = ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum',
       'Kingdom', 'Domain']#, 'immunogenicity']


final_list = []
for rank in Ranks:
  print(f'now, working on {rank}')
  final_list.append(data_distribution(data, rank, min_seq=5))  
  
print(final_list)