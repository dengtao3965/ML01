import scipy.io as sio
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
axes=fig.add_axes()

data=sio.loadmat("data/ex7data2.mat")


Kmeans=KMeans(n_clusters=3,random_state=0).fit(data['X'])
#
# print(Kmeans.labels_)
# print(Kmeans.predict([[0,0],[4,4]]))
# print(Kmeans.cluster_centers_)
index=0

fig=plt.figure()
axes=fig.add_subplot(111)
for line in data['X']:
    if Kmeans.labels_[index]==0:
        axes.scatter(line[0],line[1],color='red')
    if Kmeans.labels_[index]==1:
        axes.scatter(line[0],line[1],color='blue')
    if Kmeans.labels_[index]==2:
        axes.scatter(line[0],line[1],color='green')
    index=index+1

plt.xlabel('X')
plt.ylabel('Y')

plt.show()

