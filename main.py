import numpy as np
import matplotlib.pyplot as plt

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
#K-means Clustering[50%]
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
        old_assignment = np.array(new_assignment, copy= True)
        for i in range(np.size(x_input,0)):
            closes_centroid = 99999.9
            for j in range(k):             #assign each point to the nearest centroid
                dis = distance(x_input[i], init_centroids[j],2)
                #print("i: "+str(i)+" j: "+str(j)+" dist:"+ str(dis))
                if(dis < closes_centroid):
                    closes_centroid = dis
                    new_assignment[i] = j

        #compute new cluster centroids as mean of all points in cluster
        temp_centroid = np.zeros((k,4))
        temp_count = np.zeros(k)
        for x in range(np.size(x_input,0)):
            for y in range(k):
                if(new_assignment[x]==y):       #clusters
                    temp_centroid[y] = temp_centroid[y] + x_input[x]
                    temp_count[y] += 1
        for n in range(k):
            init_centroids[n] = np.true_divide(temp_centroid[n],temp_count[n])
        print("New centroids ")
        print(init_centroids)
        # print("new_Assignments")
        # print(new_assignment)
        # print("old_assignment")
        # print(old_assignment)
        #raw_input()

    return new_assignment, init_centroids    #left is vector with data point and cluster Assignment. right final centroids
#****************************************
#Evalutation[50%]
#****************************************
def sum_of_square_of_errors(cluster_centroids, cluster_assignments, data ):
    total_error = 0.0
    for i in range(np.size(cluster_assignments)):
        for j in range(np.size(cluster_centroids,0)):
            if(cluster_assignments[i] == j):
                total_error += distance(data[i],cluster_centroids[j],2)
    return total_error


def Knee_Plot(iris_data):
    cluster_assignments = [None]*10
    cluster_centroids = [None]*10
    errors = [None]*10
    #need to iterate through k from 1-10
    for k in range(1,11):
        print("Using K: "+str(k))
        centroids = np.array([])
        for i in range(1,k+1):
            rand_cent = np.random.randint(150.0)
            centroids = np.append(centroids,iris_data[rand_cent],axis = 0)
        centroids = centroids.reshape(k,4)
        print("Initial centroid")
        print(centroids)
        cluster_assignments[k-1], cluster_centroids[k-1] = k_means_cs171(x_input=iris_data,k=k,init_centroids=centroids)
        errors[k-1] = sum_of_square_of_errors(cluster_centroids[k-1],cluster_assignments[k-1],iris_data)
    Knee_plot = plt.figure(1)
    x = np.arange(1, 11)
    y = errors
    Knee_plot = plt.errorbar(x, y, xerr = 0, yerr=0, color = 'cornflowerblue', ecolor='crimson',capsize=1, capthick=1  )
    plt.xlabel("K = number of clusters")
    plt.ylabel("sum_of_square_of_errors")
    plt.title("Knee plot")

    plt.show()
    raw_input()


if __name__ == "__main__":
    iris_data, iris_name = Get_Data()
    Knee_Plot(iris_data)
