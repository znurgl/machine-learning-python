from numpy import *
import operator


def computeKNN(inputVector, dataSet, labels, k):
    """
    Compute the k-Nearest Neighbour of inputVector
    :param inputVector: observed vector
    :param dataSet: array of data set
    :param labels: array of labels of dataSet
    :param k:
    :return: the k-Nearest Neighbour of inputVector
    """
    dataSetSize = dataSet.shape[0]
    sqDiff = (tile(inputVector, (dataSetSize, 1)) - dataSet)**2
    distances = (sqDiff.sum(axis=1))**0.5
    distances = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[distances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

