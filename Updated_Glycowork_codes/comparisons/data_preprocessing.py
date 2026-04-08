import pandas as pd
import numpy as np
import csv
from glyles import Glycan, convert, convert_generator
from IPython.display import Image

def iupac_to_smiles(iupac_string, index):
    """
    this function converts glycan string from IUPAC -> smiles
    iupac_string: glycan string
    index: index of glycan in dataframe (just to keep the track of code)
    returns: smiles string of iupac_string"""
    #print('yes this function is running')
    glycan = Glycan(iupac_string)
    conversion_dict = {iupac_string : glycan.get_smiles()}
    if index % 1000 == 0:
        print(index)
    return conversion_dict.get(iupac_string, 'Unknown')



data = pd.read_csv('../data/v8_sugarbase.csv', sep=",", quotechar='"')
print(data)
print(data.columns)
print(data.shape)

other_unrecognized_glycans = 0
i = 0
for index, gly in enumerate(data['glycan']):
    try:
        # Attempt to create a Glycan object
        glycan = Glycan(gly, tree_only= True)
        #print("yes it works")
        pt = glycan.parse_tree

    except Exception as e:
        #print("it shows error")
        # print(f"Error processing glycan at index {index}: {e}")
        other_unrecognized_glycans += 1
        # print(index, gly)
        data = data.drop(index)

    i += 1
    if i % 1000 == 0:
        print(i)

print(data.shape)
print(other_unrecognized_glycans)

i = 0
non_convertibles = 0
for index, gly in data["glycan"].items():
    i += 1
    if i % 1000 == 0:
        print(i)
    try:
        #print(gly)
        glycan = Glycan(gly, tree_only=True)
        glycan.get_smiles()
    except Exception as e:
            #print("it shows error")
            # print(gly)
            # print(f"Error processing glycan at index {index}: {e}")
            data = data.drop(index)
            non_convertibles += 1

    
print(non_convertibles)
#print(filtered_sugarbase)
print(data.shape)

data['smiles'] = data.apply(lambda row: iupac_to_smiles(row['glycan'], row.name), axis = 1)

print(data)
print(data.shape)


new_df = data[data['smiles'] != '']
fancy_df = data[data['smiles'] == '']
print(new_df.shape)
print(fancy_df.shape)


new_df.to_csv('../data/nonrectified_iupac_to_smiles.csv',  index=False)

"""
(49587, 26)
(28463, 26)
(21124, 26)
"""