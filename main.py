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

    return new_assignment, init_centroids    #left is vector with data point and cluster Assignment. right final centroids
#****************************************
#Evalutation[50%]
#****************************************
def sum_of_square_of_errors(cluster_centroids, cluster_assignments, data ):
    total_error = 0.0
    for i in range(np.size(cluster_assignments)):
        for j in range(np.size(cluster_centroids,0)):
            if(cluster_assignments[i] == j):        #only add distance from assign centroid to point
                total_error += distance(data[i],cluster_centroids[j],2)**2  #sum up the distances from all points squared
    return total_error


def Knee_Plot(iris_data,pp):
    cluster_assignments = [None]*10
    cluster_centroids = [None]*10
    errors = [None]*10
    #need to iterate through k from 1-10
    for k in range(1,11):
        print("Using K: "+str(k))

        if(pp): #using kmean++ centroid initizliation
            centroids = kmeanspp(iris_data,k)
        else:   #using random centroids
            centroids = np.array([])
            #generate a non-repetitive random number for the centroids
            rand_cent = np.arange(150)
            np.random.shuffle(rand_cent)
            for i in range(1,k+1):
                centroids = np.append(centroids,iris_data[rand_cent[i]],axis = 0)
            #centroid in proper position
            centroids = centroids.reshape(k,4)
        print("Initial centroid")
        print(centroids)
        #
        cluster_assignments[k-1], cluster_centroids[k-1] = k_means_cs171(x_input=iris_data,k=k,init_centroids=centroids)
        errors[k-1] = sum_of_square_of_errors(cluster_centroids[k-1],cluster_assignments[k-1],iris_data)
        #print("For k: " +str(k)+" error "+str(errors[k-1]))

    return errors

def Sensitivity_analysis(iris_data, max_inter, pp):
    errors =[None]*max_inter
    for n in range(max_inter):
        errors[n] = Knee_Plot(iris_data,pp)     #get errors for the Knee_Plot y axis

    Knee_plot = plt.figure(1)                   #plot graph
    x = np.arange(1, 11)
    if(max_inter == 1):
        y = errors[0]
        yerr = 0
    else:
        y = np.mean(errors, axis = 0)
        yerr = np.std(errors, axis = 0)

    Knee_plot = plt.errorbar(x, y, xerr = 0, yerr = yerr, color = 'cornflowerblue', ecolor='crimson',capsize=2, capthick=1  )
    plt.xlabel("K = number of clusters")
    plt.ylabel("sum_of_square_of_errors")
    plt.title("Knee plot")

    plt.show()
    raw_input()

#****************************************
#K-means++ initizliation[35%]-Extra credit
#****************************************
def kmeanspp(x_input, k):
    centroid = np.array([])
    #choose first random centroid
    x = np.random.randint(0,150)
    centroid = np.append(centroid,x_input[x],axis=0).reshape(1,4)
    dist = np.zeros((150,2))
    #Get distance to all other points to the closes centroid
    for i in range(1,k):
        for j in range(np.size(x_input,0)):
            temp_dist = np.array([])
            for n in range(np.size(centroid,0)):
                temp_dist = np.append(temp_dist,distance(x_input[j],centroid[n],2)**2)
            dist[j] = [np.amin(temp_dist),j]

        all_dist = np.sum(dist[:,0])    #sum of all the distance
        dist = dist[dist[:,0].argsort()] #sort by distance and perserve index
        P = np.zeros((150,2))   #set up all the distances in a range from [0-1]
        temp_sum = 0.0
        for i in range(np.size(P,0)):
            temp_sum += dist[i][0]
            P[i] = [temp_sum / all_dist,dist[i][1]]
        #random number [0-1] that will be used to determine the index on the next centroid
        r = np.random.rand()
        for i in range(1, np.size(P,0)):
            if P[i-1][0] < r and r <= P[i][0]:
                centroid = np.append(centroid,x_input[int(P[i][1])].reshape(1,4),axis=0)
                break
    return centroid

#****************************************
#Top data points [15%]-Extra credit
#****************************************
def Top_points(iris_data,iris_name):
    #Identify the top 3 data points and match the labels
    cluster_assignments = [None]*10
    cluster_centroids = [None]*10
    errors = [None]*10
    #need to iterate through k = 3
    k = 3

    centroids = np.array([])
    #generate a non-repetitive random number for the centroids
    rand_cent = np.arange(150)
    np.random.shuffle(rand_cent)
    for i in range(1,k+1):
        centroids = np.append(centroids,iris_data[rand_cent[i]],axis = 0)
    #centroid in proper position
    centroids = centroids.reshape(k,4)
    print("Initial centroid")
    print(centroids)
    cluster_assignments, cluster_centroids = k_means_cs171(x_input=iris_data,k=k,init_centroids=centroids)
    dist1 = np.zeros((150,2))
    dist2 = np.zeros((150,2))
    dist3 = np.zeros((150,2))
    for i in range(np.size(iris_data,0)):
        dist1[i][0] = distance(iris_data[i],cluster_centroids[0],2)
        dist1[i][1] = i
        dist2[i][0] = distance(iris_data[i],cluster_centroids[1],2)
        dist2[i][1] = i
        dist3[i][0] = distance(iris_data[i],cluster_centroids[2],2)
        dist3[i][1] = i

    #sorting will give us the top three data points
    dist1 = dist1[dist1[:,0].argsort()]
    dist2 = dist2[dist2[:,0].argsort()]
    dist3 = dist3[dist3[:,0].argsort()]
    #Print top 3 class labels for the 3 clusters

    print("Cluster: 1")
    print(iris_name[int(dist1[0][1])])
    print(iris_name[int(dist1[1][1])])
    print(iris_name[int(dist1[2][1])])

    print("Cluster: 2")
    print(iris_name[int(dist2[0][1])])
    print(iris_name[int(dist2[1][1])])
    print(iris_name[int(dist2[2][1])])

    print("Cluster: 3")
    print(iris_name[int(dist3[0][1])])
    print(iris_name[int(dist3[1][1])])
    print(iris_name[int(dist3[2][1])])

if __name__ == "__main__":
    iris_data, iris_name = Get_Data()
    #**************How to run*****************************
    #First parameter data to cluster
    #Second parameter number of iterations over data
    #Third parameter bool pp use K-means++ if true else random initizliation
    Sensitivity_analysis(iris_data,1,pp = False)



    Top_points(iris_data,iris_name)
