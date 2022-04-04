import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_mapping(col):
    mapping = {}
    for idx, data in enumerate(col.unique()):
        mapping[data] = idx
    return mapping


def process_data(data):
    brand_mapping = get_mapping(data['brand'])
    model_mapping = get_mapping(data['model'])
    dummy_cols = ['transmission', 'fuelType']
    temp = data.replace(brand_mapping)
    temp = temp.replace(model_mapping)
    temp = pd.get_dummies(temp, columns=dummy_cols)
    return temp


x_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('test_label/y_test.csv')

x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')

x_data = process_data(x_train)
p_data = PCA(2).fit_transform(x_data)

kmeans = KMeans(n_clusters=12, random_state=0).fit(x_data)
labels = kmeans.fit_predict(p_data)
u_labels = np.unique(labels)

for i in u_labels:
    f_label = p_data[labels==i]
    plt.scatter(p_data[labels == i, 0], p_data[labels == i, 1], label = i)
plt.legend()
plt.show()