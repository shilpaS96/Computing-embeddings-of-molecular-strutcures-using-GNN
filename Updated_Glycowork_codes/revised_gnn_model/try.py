import pandas as pd

# Sample data
data = {
    'glycan': ['GalNAc', 'Fuc'],
    'domains': [
        ['Bacteria', 'Bacteria', 'Bacteria', 'Bacteria', 'Bacteria'],
        ['Bacteria', 'Bacteria', 'Bacteria', 'Animalia', 'Bacteria']
    ]
}
df = pd.DataFrame(data)

# Explode and reset index
df_exploded = df.explode('domains').reset_index(drop=True)

print(df_exploded)

import pandas as pd

# Sample DataFrame
data = {
    'glycan': ['GalNAc', 'GalNAc', 'GalNAc', 'GalNAc', 'GalNAc', 'Fuc', 'Fuc', 'Fuc', 'Fuc', 'Fuc'],
    'domains': ['Bacteria', 'Bacteria', 'Bacteria', 'Bacteria', 'Bacteria', 'Bacteria', 'Bacteria', 'Bacteria', 'Animalia', 'Bacteria'],
    'smiles': ['S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2', 'S2']
}
df = pd.DataFrame(data)

# Remove duplicates based on 'glycan' and 'domains' columns only
df_unique = df.drop_duplicates(subset=['glycan', 'domains']).reset_index(drop=True)

print(df_unique)