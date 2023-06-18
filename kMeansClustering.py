#https://archive-beta.ics.uci.edu/dataset/53/iris
import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import NearestCentroid


def getCol(clusters, i):
    col = []
    for j in range(1 , clusters[i].shape[0]):
        col.append(clusters[i][j])
    return col

def addVectors(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result


#euclidian distance between two data points
def distance (a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def objectiveFunction (clusters, means):
    result = 0
    for k in range(len(means)):
        for n in range(1, len(clusters[k])):
            result = result + math.pow(distance(clusters[k][n], means[k]), 2)
    return result

def learningProcessPlot (data, means, stage):
    xClusters = []
    for i in range(data.shape[0]):
        xClusters.append(data[i][2])
    yClusters = []
    for i in range(data.shape[0]):
        yClusters.append(data[i][3])
    xMeans = []
    for i in range(len(means)):
        xMeans.append(means[i][2])
    yMeans = []
    for i in range(len(means)):
        yMeans.append(means[i][3])
    plt.scatter(xClusters, yClusters, c = "black", edgecolors= "red")
    plt.scatter(xMeans, yMeans, c = "yellow", marker= "s", edgecolors= "black")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title(stage)
    plt.show()

def getClassification(point, means, meanL, meanW):
    #data = [0, 0, data[0], data[1]]
    closestMean = 0
    for k in range(1, len(means)):
        if distance(np.array([meanL, meanW, point[0], point[1]]), means[k]) < distance(np.array([meanL, meanW, point[0], point[1]]), means[closestMean]):
            closestMean = k
    return closestMean


def decisionBoundary(data, labels, k):
    colors = ("green", "red", "blue")
    colorMap = ListedColormap(colors[:k])
    nC = NearestCentroid()
    classifier = nC.fit(data, labels)

    x1Min = data[:,2].min() - 1
    x1Max = data[:,2].max() + 1
    x2Min = data[:,3].min() - 1
    x2Max = data[:,3].max() + 1
    meanL = np.mean(data[:, 0])
    meanW = np.mean(data[:,1])
    xx1, xx2 = np.meshgrid(np.arange(x1Min, x1Max, 0.2), np.arange(x2Min, x2Max, 0.2))
    xx1Arr = xx1.ravel()
    xx2Arr = xx2.ravel()
    meanL = [meanL for i in range(len(xx1Arr))]
    meanW = [meanW for i in range(len(xx2Arr))]

    predictions = classifier.predict(np.array([meanL, meanW, xx1Arr, xx2Arr]).T)
    predictions = predictions.reshape(xx1.shape)

    plt.contourf(xx1, xx2, predictions, alpha=0.4, cmap=colorMap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xlabel('Petal length')
    plt.ylabel('Petal Width')
    plt.show()

    

def objectiveFunctionPlot(dist):
    x = []
    for i in range(len(dist)):
        x.append(i)
    plt.scatter(x, dist, c = "blue")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function")
    plt.show()


#k = number of clusters. data = all of given data
def kMeansClustering(k, data):
    means = []
    #get k random means
    for i in range(k):
        means.append(data[random.randint(0, data.shape[0] - 1)])
    clusters = [[0 for i in range(1)] for j in range(k)]
    objectiveFunctionPerIteration = []
    learningProcessPlot(data, means, "Starting Graph")
    for x in range(100):
        if x == 50:
            learningProcessPlot(data, means, "Intermediate Graph")
        #reset clusters to empty
        clusters = [[0 for i in range(1)] for j in range(k)]
        #assign each data point into its cluster
        for i in range(0, data.shape[0]):
            closestMean = 0
            for j in range(1, len(means)):
                if distance(means[closestMean], data[i]) > distance(means[j], data[i]):
                    #found a closer mean
                    closestMean = j
            clusters[closestMean].append(data[i])
        #All data points are organized into their clusters
        #update the means one coordinate at a time
        for i in range(k):
            newMean  =[0,0,0,0]
            for j in range(1, len(clusters[i])):
                newMean = addVectors(newMean, clusters[i][j])
            #scale the mean by the number of data entries
            for j in range(len(newMean)):
                newMean[j]= newMean[j] / (len(clusters[i]) - 1)
            means[i] = newMean
        objectiveFunctionPerIteration.append(objectiveFunction(clusters, means))
    #All iterations terminated. Generating Final Plots...
    objectiveFunctionPlot(objectiveFunctionPerIteration)
    learningProcessPlot(data, means, "Final Graph")

if __name__ == '__main__':
    dataFrame = pd.read_csv(r"C:\Users\benku\College\2022-23\AI\Project 2\iris.csv")
    data = dataFrame[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]].to_numpy()
    labels = dataFrame[["NumClass"]].to_numpy()
    kMeansClustering(3,data)
    decisionBoundary(data, labels, 3)
    