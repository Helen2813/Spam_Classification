import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('spam.tsv', sep='\t')

# df.isna().sum()

# balancing data
ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']

ham = ham.sample(spam.shape[0])
data = pd.concat([ham, spam], ignore_index=True)
print(data.shape)