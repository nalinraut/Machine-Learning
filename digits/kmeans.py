import random
import math
import matplotlib.pyplot as plot
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


def readData():
	numbers = load_digits(n_class=10)
	data = numbers.data
	dimensions = len(data[0])
	gt_Labels = numbers.target
	return (data, dimensions, gt_Labels)


def getDistancesFromCentroids(dataItem, centroids):

	dists = []

	for centroid in centroids:
		
		dist = 0
		for i in range(len(dataItem)):
			a = dataItem[i]-centroid[i]
			dist += a*a

		dists.append(math.sqrt(dist))

	return dists


def kmeans(unclustered_data, dimensions, n_clusters):
	
	data = unclustered_data

	dataClusterPredictions = [-1 for i in range(len(data))]

	centroids = list(data[0:10])
	prevCentroids = []

	while True:

		clusters = [[] for i in range(n_clusters)]


		for i in range(len(data)): 
			distances = getDistancesFromCentroids(data[i], centroids)
			clusterNumber = distances.index(min(distances))

			clusters[clusterNumber].append(data[i])
			dataClusterPredictions[i]=clusterNumber

		centroids = [[0 for i in range(dimensions)] for j in range(n_clusters)]
		
		for j in range(n_clusters):
			
			dimensionSums = []
			for k in range(dimensions):
				dimensionSum = 0
				for ele in clusters[j]:
					dimensionSum += ele[k]
				dimensionSums.append(dimensionSum)

			for k in range(len(dimensionSums)):
				centroids[j][k] = float(dimensionSums[k])/len(clusters[j])

		if centroids==prevCentroids:
			return (clusters,dataClusterPredictions)

		prevCentroids = list(centroids)
		

def analyseClusters(gt_Labels, dataClusterPredictions):

	clusterwiseDigitCounts = [[0 for i in range(10)] for j in range(10)]

	for i in range(len(dataClusterPredictions)):
		clusterNumber = dataClusterPredictions[i]
		digit = gt_Labels[i]
		clusterwiseDigitCounts[clusterNumber][digit] += 1


	print ("\nCluster digit counts and cluster labels : ")

	print("0s 1s 2s 3s 4s 5s 6s 7s 8s 9s")
	labels_clust=[]
	confusionMatrixCalculations = []
	for digitCount in clusterwiseDigitCounts:
		labels_clust.append(digitCount.index(max(digitCount)))
		print (digitCount,"\t",digitCount.index(max(digitCount)))
		confusionMatrixCalculations.append((digitCount.index(max(digitCount)), digitCount))


	confusionMatrixCalculations.sort()
	print ("\nConfusion Matrix : ")

	#printing Confusion Matrix 
	cf=[]
	for i in range(len(confusionMatrixCalculations[0][1])):
		cf.append(confusionMatrixCalculations[i][1])

	print(np.array(cf).T)




	actualClusterPredictions = []

	for dataClusterPrediction in dataClusterPredictions:
		actualClusterPredictions.append(labels_clust[dataClusterPrediction])


	fmi = metrics.fowlkes_mallows_score(gt_Labels, actualClusterPredictions)

	print ("\nThe Fowlkes-Mallow index is :",fmi)


data, dimensions,gt_Labels= readData()

clusters, dataClusterPredictions = kmeans(data, dimensions, 10)

analyseClusters(gt_Labels, dataClusterPredictions)