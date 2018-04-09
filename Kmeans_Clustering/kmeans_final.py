
# Nalin Yatin Raut
import os
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas
import random

# File and data extraction
data = pandas.read_table('realdata.txt')
X = np.array(data)
# Color array 
colors = ["c", "m","r", "g", "b", "y"]
names = ["Cluster 1", "Cluster 2","Cluster 3","Cluster 4","Cluster 5","Cluster 6"]
new_data = np.array(data)

c = 0

# Plotting Original Data
for x in X:
    new_data[c][0] = x[1]
    new_data[c][1] = x[2]
    c = c+1
    plt.scatter(x[1], x[2], marker = "x", color = colors, s = 15, linewidth = 1)
plt.title('UNCLUSTERED DATA')
plt.xlabel('LENGTH')
plt.ylabel('WIDTH')        
plt.show()
np.random.shuffle(new_data)

# K-Mean Algorithm
class K_Means:
    
    # Initialization 
    def _init_(self, k, it):
        self.k = k
        self.tol = tol
        self.it = 400
    
    # Calculate Centroids  
    def data_fitting(self, data, k, it):
        self.centroids = {}
        
        for i in range(k):
            self.centroids[i] = new_data[i] #selecting first two points of the data as centroids
          
        for i in range(it):
            self.classifications = {}
            
            for i in range(k):
                self.classifications[i] = []
                
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(features)
            
            prev_centroids = dict(self.centroids)
                
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis = 0 )
            
            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/ original_centroid * 100.0) > .001:
                    optimized = False
                
                if optimized:
                    break    
               
    def predict(self, data):
        distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]# Calculating distances of features from centroid
        classification = distances.index(min(distances))# picking the minimum distance from all the distances
        return classification   



def main():

    k=int(input('Enter the number of clusters between 1 and 6 you want: '))

    classifier = K_Means()
    classifier.data_fitting(new_data, k, 1000)

# Plot Clustered Data
    c1_plot=[]
    c2_plot=[]
    for centroid in classifier.centroids:
        c1=classifier.centroids[centroid][0]
        c1_plot.append(c1)
        c2=classifier.centroids[centroid][1]
        c2_plot.append(c2)
    
    fig, ax = plt.subplots()
    ax.scatter(c1_plot, c2_plot, color='k', s=15, linewidth = 10)
    for i, txt in enumerate(names):
        if i==len(c1_plot):
            break
        ax.annotate(txt, (c1_plot[i],c2_plot[i]))

    
    
        #ax.scatter(classifier.centroids[centroid][0], classifier.centroids[centroid][1], color = "k", s = 15, linewidth = 10, label=names)

    
    for classification in classifier.classifications:
        color = colors[classification]
        for features in classifier.classifications[classification]:
            plt.scatter(features[0], features[1], marker = "x", color = color, s = 15, linewidth = 1)

  

    plt.title('CLUSTERED DATA')
    plt.xlabel('LENGTH')
    plt.ylabel('WIDTH')
    plt.show()
main()





