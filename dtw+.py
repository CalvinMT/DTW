from __future__ import print_function

from dtw import runSearch
from dtw import computeResults
from dtw import showSweeps
from dtw import printStatistics

import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse

GRAPH = False
VERBOSE = False

def getMedian(bestList, threshold):
    # FIXME - should not use bestList
    """
    TODO

    :param bestList:
    :param threshold:
    :return:
    """
    sortedArray = sorted(bestList, key=lambda x: x[1])
    sortedIndeces = np.argsort(np.array(bestList)[:, 1]).tolist()
    i = 0
    while i < len(sortedArray):
        if sortedArray[i][1] > threshold:
            sortedArray.pop(i)
            sortedIndeces.pop(i)
            i -= 1
        i += 1
    fileIndex = sortedIndeces[len(sortedIndeces) / 2]
    sweepIndex = sortedArray[len(sortedArray) / 2][0]
    median = sortedArray[len(sortedArray) / 2][1]
    return fileIndex, sweepIndex, median

def getNewQueryPath(fileIndex, searchPatternPath):
    """
    TODO

    :param fileIndex:
    :param searchPatternPath:
    :return: path to query audio
    """
    searchFileList = sorted(glob.glob(searchPatternPath))
    newQueryPath = searchFileList[fileIndex]
    return newQueryPath

def mergeResults(previousResults, newResults):
    """
    TODO

    :param previousResults:
    :param newResults:
    :return:
    """
    mergedResults = []
    for i in range(len(previousResults)):
        p = 0
        n = 0
        mergedResults.append([])
        while p < len(previousResults[i]) or n < len(newResults[i]):
            if n >= len(newResults[i]) or (p < len(previousResults[i]) and previousResults[i][p][0] < newResults[i][n][0]):
                mergedResults[i].append(previousResults[i][p])
            elif p >= len(previousResults[i]) or (n < len(newResults[i]) and previousResults[i][p][0] > newResults[i][n][0]):
                mergedResults[i].append(newResults[i][n])
                p -= 1
                n += 1
            else:
                if previousResults[i][p][1]:
                    mergedResults[i].append(previousResults[i][p])
                else:
                    mergedResults[i].append(newResults[i][n])
                n += 1
            p += 1
    return mergedResults

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dynamic Time Warping')
    parser.add_argument('-g', '--graph', action='store_true', help='Enable graph display')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Set score threshold')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')
    parser.add_argument('query_path')
    parser.add_argument('search_pattern_path')
    args = parser.parse_args()

    GRAPH = args.graph
    threshold = args.threshold
    VERBOSE = args.verbose
    queryPath = args.query_path
    searchPatternPath = args.search_pattern_path

    labels, sweepList, bestList = runSearch(queryPath, searchPatternPath)
    results = computeResults(sweepList, threshold, oneWord=True)

    for i in range(3):
        fileIndex, sweepIndex, median = getMedian(bestList, threshold)
        queryPath = getNewQueryPath(fileIndex, searchPatternPath)

        labels, sweepList, bestList = runSearch(queryPath, searchPatternPath)
        newResults = computeResults(sweepList, threshold, oneWord=True)

        results = mergeResults(results, newResults)

    if GRAPH:
        showSweeps(labels, sweepList, bestList)

    ####################################################
    # XXX - Predictions
    #
    predictions = [[[0, 1]]] * len(results)
    #
    #predictions = [[[0, 0]]] * 1664 # backward
    #predictions = predictions + [[[0, 1]]] * 2014 # bed
    #predictions = predictions + [[[0, 0]]] * 2064 # bird
    #
    ####################################################

    printStatistics(results, predictions)

    plt.show()
