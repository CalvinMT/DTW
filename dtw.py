from __future__ import print_function

from speech_dtw import qbe
from python_speech_features import mfcc
from python_speech_features import delta

import utils

import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import operator

PERCENTAGE = False
VERBOSE = False

def getMFCC(rate, signal):
    """
    TODO
    """
    mel = mfcc(signal, rate) # x: time, y: coefficients
    delta1 = delta(mel, 2) # speed
    delta2 = delta(delta1, 2) # acceleration
    features = np.hstack([mel, delta1, delta2])
    cmvn = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return cmvn

def runSearch(queryPath, searchPatternPath="", searchPathList=None):
    """
    Search for instances of a given audio query in multiple audio files using DTW.
    Example:
        queryPath - bed/query.wav
        searchPatternPath - b*d/???????_search.wav
    Returning labels are the search filenames (1D).
    Returning sweepList is the sweeping result per search file (2D).
    Returning bestList is the tuple (index, score) of best match per search file (2D).

    :param queryPath: path to a single audio file
    :param searchPatternPath: path to multiple audio files
    :return: labels, sweepList, bestList
    """
    # Search filenames (1D)
    labels = []
    # Sweeping results per search file (2D)
    sweepList = []
    # Index and score of best match per search file (2D)
    bestList = []

    # Compute query's MFCC
    (qRate, query) = wav.read(queryPath)
    queryMFCC = getMFCC(qRate, query)

    if searchPathList == None:
        # List all search filenames from search pattern
        searchFileList = sorted(glob.glob(searchPatternPath))
    else:
        searchFileList = searchPathList
    for i, searchFile in enumerate(searchFileList):
        labels.append(searchFile.split("/")[-1])
        if VERBOSE:
            folderName = searchFile.split("/")[-2]
            print("Searching '" + labels[-1] + "' in " + folderName + "...", end='')
        # Compute search file's MFCC
        (sRate, search) = wav.read(searchFile)
        searchMFCC = getMFCC(sRate, search)
        # Sweep query across search file
        sweepList.append(qbe.dtw_sweep(queryMFCC, searchMFCC, 3)) # x: time/3, y: cost
        if VERBOSE:
            print(" Done")
        if PERCENTAGE:
            print("%.2f" % (i * 100 / len(searchFileList)) + "%", end='\r')
    
    bestList = [[0, 0] for i in range(len(sweepList))]
    # Find best match for each search file
    for i, sweep in enumerate(sweepList):
        bestList[i][0], bestList[i][1] = min(enumerate(sweep), key=operator.itemgetter(1))

    return labels, sweepList, bestList

def computeResults(sweepList, threshold, positiveOnly=False, oneWord=False):
    """
    Compare sweeping scores with the given threshold. The result is 0 if the score is 
    strictly higher than the threshold; 1 otherwise.

    :param sweepList: sweep array from search
    :param threshold: float between 0.0 and 1.0
    :param positiveOnly: boolean, True to return only positive results
    :param oneWord: boolean, True if query and search files are made of one word only
    :return: array with (sweep_index, result) for each search file
    """
    # results = [sweep_index, 0 or 1]
    results = []
    for i, sweep in enumerate(sweepList):
        results.append([])
        for j, score in enumerate(sweep):
            if score <= threshold:
                if oneWord:
                    results[i].append([0, 1])
                    break
                else:
                    results[i].append([j, 1])
            else:
                if not positiveOnly and not oneWord:
                    results[i].append([j, 0])
                elif not positiveOnly and oneWord and j == len(sweep) - 1:
                    results[i].append([0, 0])
    return results

def showSweeps(labels, sweepList, bestList, y_min=0.0, y_max=1.0):
    """
    Prepare a figure to show sweeping results for all search files. The best score per 
    search file is highlited.

    :param labels: array of strings
    :param sweepList: sweep array from search
    :param bestList: best match array from search
    :param y_min: minimum of y axis
    :param y_max: maximum of y axis
    :return: None
    """
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([y_min, y_max])
    for i, sweep in enumerate(sweepList):
        color = next(axes._get_lines.prop_cycler)['color']
        plt.plot(sweep, ':', color=color, zorder=i)

    if VERBOSE:
        print("")
        print("Best match (position, score)")
    axes.set_prop_cycle(None)
    for i, best in enumerate(bestList):
        color = next(axes._get_lines.prop_cycler)['color']
        plt.plot(best[0], best[1], 'o', color=color, zorder=i+len(sweepList)-1)
        if VERBOSE:
            rgb = utils.hexToRGB(color)
            print(utils.get_color_escape(255, 255, 255, True), end='')
            print(utils.get_color_escape(rgb[0], rgb[1], rgb[2]), end='')
            print(labels[i] + ": " + str(best), end='')
            print(utils.RESET_COLOR_ESCAPE)

    axes.set_xlabel('Time', fontsize='x-large')
    axes.set_ylabel('Score', fontsize='x-large')
    axes.legend(labels)
