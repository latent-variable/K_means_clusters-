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
    dist= 0;
    for i in range(len(x)):
        dist+= (abs(x[i] -y[i]))**p
    return((dist)**(1.0/p))

def k_means_cs171(x_input,           #this is a datapoint x feature) matrix. Each row contains a data point to be clustered.
                  k,                #this is the number of clusters
                  init_centroids): #this is a (k* feature ) matrix with the initizliation for the k init_centroids

    old_assignment = np.zeros(150)
    new_assignment = np.arange(150)
    while(not(np.array_equal(old_assignment, new_assignment))):     #repeat while clustering changes
        old_assignment = new_assignment
        for i in range(np.size(x_input,0)):
            closes_centroid = 99999.9
            for j in range(k):             #assign each point to the nearest centroid
                dis = distance(x_input[i], init_centroids[j],2)
                #print("i: "+str(i)+" j: "+str(j)+" dist:"+ str(dis))
                #raw_input()
                if(dis < closes_centroid):
                    closes_centroid = dis
                    new_assignment[i] = j


        print(new_assignment)
        #compute new cluster centroids as mean of all points in cluster
        temp_centroid = np.zeros((k,4))
        temp_count = np.zeros(k)
        print(temp_centroid)
        for x in range(np.size(x_input,0)):
            for y in range(k):
                if(new_assignment[x]==y):       #clusters
                    temp_centroid[y] = temp_centroid[y] + x_input[x]
                    temp_count[y] += 1
        print(temp_count)
        for n in range(k):
            init_centroids[n] = np.true_divide(temp_centroid[n],temp_count[n])

        print(init_centroids)
        raw_input

    return new_assignment, init_centroids    #left is vector with data point and cluster Assignment. right final centroids

if __name__ == "__main__":
    iris_data, iris_name = Get_Data()

    #need to iterate through k from 1-10
    k = 3
    centroids = np.array([])
    for i in range(1,k+1):
        rand_cent = i*np.random.randint(150.0/k)
        print("cent: "+str(rand_cent))
        centroids = np.append(centroids,iris_data[rand_cent],axis = 0)
    centroids = centroids.reshape(3,4)
    #print(centroids)
    cluster_assignments, cluster_centroids = k_means_cs171(x_input=iris_data,k=3,init_centroids=centroids)
