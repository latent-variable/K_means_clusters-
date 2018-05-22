import numpy as np
import matplotlib as plt

#****************************************
#Gettting real data [0%]
#****************************************
#Working with Iris_data set
def Get_Data():
    fname = 'iris_data.txt'
    data = np.loadtxt(fname, delimiter=',', usecols = (0,1,2,3)) #Removed instances of missing data
    data_name = np.loadtxt(fname, delimiter=',', dtype=str, usecols = (4))
    data = data.reshape(150,4)
    data_name = data_name.reshape(150,1)
    return data,data_name

#****************************************
#K-means Clustering[0%]
#****************************************
def distance(x,y,p):                #Lp distance using L2 distance for this Assignment
    distance = 0;
    for i in range(len(x)):
        distance += (abs(x[i] -y[i]))**p
    return((distance)**(1.0/p))

def k_means_cs171(x_input,           #this is a datapoint x feature) matrix. Each row contains a data point to be clustered.
                  k,                #this is the number of clusters
                  init_centroids): #this is a (k* feature ) matrix with the initizliation for the k init_centroids


    return cluster_assignments, cluster_centroids    #left is vector with data point and cluster Assignment. right final centroids

if __name__ == "__main__":
    iris_data, iris_name = Get_Data()

    #need to iterate through k from 1-10
    k = 3
    centroids = []
    for i in range(k):
        centroids.append( i*iris_data[np.random.randint(150.0/k)])
        
    cluster_assignments, cluster_centroids = k_means_cs171(x_input=iris_data,k=3,init_centroids=centroids)
    print(iris_name)
