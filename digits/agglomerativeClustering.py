'''Nalin Yatin Raut
AI Assignment Problem 3 Part B'''

from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np

def extractData(numberClass):
	'''extracts data from sklearn datasets'''
	number=load_digits(n_class=numberClass)
	data=number.data
	gt_labels= number.target #ground truth
	return(number, data, gt_labels)


def agglomerativeClustering(num,d, gt):
	'''Computes clusters, its labels and confusion matrix with
	fowlkes_mallows index .'''
	gt_labels=gt
	number=num
	data=d
	countsInitial = [0]*10
	for k in gt_labels:
		countsInitial[k] += 1

	print("Number of samples:")
	for i in range(10):
		print ("For digit "+str(i)+", the count is: "+str(countsInitial[i]))

	clustering = AgglomerativeClustering(n_clusters=10,linkage='ward')
	labels = clustering.fit_predict(data)

	clusterwiseDigitCounts = [[0 for i in range(10)] for j in range(10)]

	for i in range(len(labels)):
		clusterNumber = labels[i]
		digit = gt_labels[i]
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

	truepredictions=[]

	for i in range(len(labels)):
		truepredictions.append(labels_clust[labels[i]])
	
	fm_score = metrics.fowlkes_mallows_score(gt_labels, truepredictions)
	print ("\nThe Fowlkes-Mallow's index is :",fm_score)

def main():
	classNumber=10
	n,d,g=extractData(classNumber)
	agglomerativeClustering(n,d, g)
main()
