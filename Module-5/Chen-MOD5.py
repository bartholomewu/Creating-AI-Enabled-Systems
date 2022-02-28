import pandas as pd
import numpy as np
from sklearn import preprocessing


def get_str_columns(data):
    cols = data.columns
    str_cols = []
    for col in cols:
        if isinstance(data[col].loc[0], str):
            print(col)
            print(data[col].unique())
            print()
            str_cols.append(col)
    return str_cols


label_encoder = preprocessing.LabelEncoder()

data = pd.read_csv('cars.csv')

str_columns = get_str_columns(data)

data_dummies = pd.get_dummies(data, columns = str_columns)
print(data_dummies)

# Data Categorization via One-Hot-Encoding.
# The size of the dataset multiplied quite extensively from (38531,30) to (38531,1240). While this makes for the easiest processing (everything is either a set value or a binary value), it strains the memory usage as well as makes it harder for us to read (due to the large size of the data).

for col in str_columns:
    data[col] = label_encoder.fit_transform(data[col])
print(data)

# Data Categorization via Label Encoding.
# The size of the data has roughly slimmed down, however manual reading of the dataset requires the usage of a legend/mapping. While the data has been transformed to be easily readible to machines, the end output needs to be decoded from the initial encoding.