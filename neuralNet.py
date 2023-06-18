import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient(x, bias, weights, label, point):
    dotProduct = bias + np.dot(weights, point)
    res = (label - 1 - sigmoid(dotProduct)) * math.pow((1 + np.exp(-dotProduct)),-2) * np.exp(-dotProduct) * -point[x]
    return res

def decisionBoundary(data, weights, bias):
    #below tolerance
    x1 = []
    y1 = []
    #above tolerance
    x2 = []
    y2 = []
    for index in range(data.shape[0]):
        dotProduct = np.dot(weights, data[index]) + bias
        yHat = sigmoid(dotProduct)
        if yHat < 0.5:
            x1.append(data[index][2])
            y1.append(data[index][3])
        if yHat > 0.5:
            x2.append(data[index][2])
            y2.append(data[index][3])
    fig, ax = plt.subplots()
    slope = (max(y2) - min(y1)) / (max(x2) - min(x1))
    y_intercept = min(y1) - slope * min(x1)
    line_x = [min(x1), max(x2)]
    line_y = [slope * x + y_intercept for x in line_x]

    plt.scatter(x1, y1, color='red')
    plt.scatter(x2, y2, color='blue')
    ax.plot(line_x, line_y, color='green')
    plt.show()

def calculate_mean_squared_err(data, labels, parameters):
    weights = parameters[1:5]
    bias = parameters[0]
    meanSquaredErr = 0
    for index in range(data.shape[0]):
        dotProduct = np.dot(weights, data[index]) + bias
        yHat = sigmoid(dotProduct)
        meanSquaredErr += math.pow(labels[index] - 1 - yHat, 2)
    meanSquaredErr = meanSquaredErr / (data.shape[0])
    print("Mean Squared Error = ")
    print(meanSquaredErr)
    decisionBoundary(data, weights, bias)

def gradientDescent(data, labels, parameters):
    weights = parameters[1:5]
    weights = np.array(weights, dtype = np.float64)
    bias = parameters[0]
    alpha = 1
    summedGradient = 0
    for i in range (len(weights)):
        loss = 0
        for j in range (data.shape[0]):
            loss += gradient(i, bias, weights, labels[j][0], data[j])
        loss = loss * 2 / data.shape[0]
        weights[i] = weights[i] - alpha * loss
        summedGradient += loss
    print("Weights after gradient descent = ")
    print(weights)
    print("Summed Gradient = ")
    print(summedGradient)

if __name__ == '__main__':
    dataFrame = pd.read_csv(r"C:\Users\benku\College\2022-23\AI\Project 2\iris.csv")
    data = dataFrame[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]].to_numpy()
    labels = dataFrame[["NumClass"]].to_numpy()
    parameters = np.array([0,-2.83021573, -0.77314731,  4.55262142, -1.28904998])
    #data restricted to only the last two classes is called gradientData because the gradient should only be calculated using the last two classes
    gradientData = data[50:152]
    gradientLabels = labels[50: 152]
    calculate_mean_squared_err(gradientData, gradientLabels, parameters)
    gradientDescent(gradientData, gradientLabels, parameters)

#Low MSE = [0,-2.8,-0.79,4.52 ,-1.3]
#High MSE = [0, -4, 2, 5, -4]