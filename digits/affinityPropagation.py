'''Nalin Yatin Raut
AI Assignment Problem 3 Part C'''

from sklearn.datasets import load_digits
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np

def extractData(numberClass):
	'''extracts data from sklearn datasets'''
	number=load_digits(n_class=numberClass)
	data=number.data
	gt_labels= number.target #ground truth
	return(number, data, gt_labels)

def affinityPropagation(num,d, gt):
	'''Computes clusters, its labels and confusion matrix with
	fowlkes_mallows index .'''
	gt_labels=gt
	number=num
	data=d
	countsInitial = [0]*10
	for k in gt_labels:
		countsInitial[k]+=1

	print('Count for digits from (0 to 9)')
	for i in range(10):
		print('For digit'+str(i)+', the count is: '+str(countsInitial[i]))
	clustering = AffinityPropagation(damping=0.9, preference= -70000, affinity='euclidean')
	labels = clustering.fit(data)

	print('Number of Clusters: ', len(labels.cluster_centers_))
	print('Number of Iterations: ', labels.n_iter_)
	data_labels=labels.labels_

	clusterDigitCount=[[0 for i in range(10)] for j in range(10)]
	for i in range (len(data_labels)):
		clusterDigitCount[data_labels[i]][gt_labels[i]]+=1

	print('\n Cluster digit count, and labels:')
	print("0s 1s 2s 3s 4s 5s 6s 7s 8s 9s")
	labels_clust=[]
	confusionMatrixCalculations = []
	for digitCount in clusterDigitCount:
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

	for i in range(len(data_labels)):
		truepredictions.append(labels_clust[data_labels[i]])

	fm_score = metrics.fowlkes_mallows_score(gt_labels, truepredictions)

	print ("\nThe Fowlkes-Mallow score is :",fm_score)

def main():
	classNumber=10
	n,d,g=extractData(classNumber)
	affinityPropagation(n,d, g)
main()





