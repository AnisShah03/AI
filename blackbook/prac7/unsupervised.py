import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

customer_data = pd.read_csv('BlackBookPracAAI\prac7\Mall_Customers.csv')
# print(customer_data)
customer_data.shape
customer_data.head()
data = customer_data.iloc[:,3:5].values
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title("Customer dendogram")

dend = shc.dendrogram(shc.linkage(data,method='ward'))
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
cluster.fit_predict(data)
plt.figure(figsize=(10,7))
plt.scatter(data[:,0],data[:,1],c=cluster.labels_,cmap='rainbow')
plt.show()


