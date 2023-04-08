#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
#%%
data = pd.read_csv("data.csv")

#%%
data.head()
data.describe()
data.info()
plt.figure(figsize=(10,7))
sns.boxplot(data=data.iloc[:,2:4])
plt.show()

#%%
x=data.iloc[:,2:4]
x.head()
#%%
#outliers
x=data.iloc[:,2:4]
cols = ['Milk', 'Fresh']
for col in cols:
  q1 = x[col].quantile(0.25)
  q3 = x[col].quantile(0.75)
  iqr = q3-q1
  min = q1 - 1.5 * iqr
  max = q3 + 1.5 * iqr
  for y in range(0, len(x[col])):
    value = x[col][y] 
    if value  > max:
        value  = max
    if value  < min:
        value = min
    x[col][y] = value 

    
plt.figure(figsize=(15,10))
sns.boxplot(data=x)
plt.show()
#%%
#scale
scaler=MinMaxScaler()
x = scaler.fit_transform(x)
plt.figure(figsize=(10,7))
sns.boxplot(data=x)
plt.show()
#%%

import matplotlib.pyplot as mtp
from sklearn.cluster import KMeans
wcss_list= []  #Initializing the list for the values of WCSS

for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state= 42)
    km.fit(x)
    wcss_list.append(km.inertia_)
    
mtp.plot(range(1, 11), wcss_list)
mtp.title('The Elbow Method Graph')
mtp.xlabel('Number of Clusters, K')
mtp.show()

# %%
km = KMeans(n_clusters=6, random_state= 42)
label= km.fit_predict(x)
print(label)

# %%

#plotting the results
plt.scatter(x[:,0], x[:,1], c=label, s=50, cmap='viridis')
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Fresh')
plt.ylabel('Milk')
plt.show()
# %%
from sklearn.cluster import DBSCAN
cluster = DBSCAN(eps=0.1, min_samples=2)
cluster.fit(x)
print(f'DBSCAN found {len(set(cluster.labels_) - set([-1]))} clusters and {(cluster.labels_ == -1).sum()} points of noise.')
# %%
plt.rcParams['figure.figsize'] = (20,15)
unique_labels = set(cluster.labels_)
n_labels = len(unique_labels)
cmap = plt.cm.get_cmap('brg', n_labels)
for l in unique_labels:
    plt.scatter(
        x[cluster.labels_ == l, 0],
        x[cluster.labels_ == l, 1],
        c=[cmap(l) if l >= 0 else 'Black'],
        marker='ov'[l%2],
        alpha=0.75,
        s=100,
        label=f'Cluster {l}' if l >= 0 else 'Noise')
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
# %%
from sklearn.cluster import AgglomerativeClustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(x)
plt.scatter(x[:,0], x[:,1], c=labels, s=50, cmap='viridis')
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Fresh')
plt.ylabel('Milk')
plt.show()
# %%
