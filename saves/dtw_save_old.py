from __future__ import print_function

from speech_dtw import qbe
from python_speech_features import mfcc
from python_speech_features import delta

import matplotlib.pyplot as plt
import glob
import numpy as np
import scipy.io.wavfile as wav
import operator
import argparse

VERBOSE = False

# ---------- --COLOUR-- ----------
RESET_COLOR_ESCAPE = '\033[0m'
def get_color_escape(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)
def hexToRGB(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
# ---------- ---------- ----------

def getMFCC(rate, signal):
    mel = mfcc(signal, rate) # x: time, y: coefficients
    delta1 = delta(mel, 2) # speed
    delta2 = delta(delta1, 2) # acceleration
    features = np.hstack([mel, delta1, delta2])
    cmvn = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return cmvn

def runSearch(queryPath, searchPatternPath):
    # Search filenames (1D)
    labels = []
    # Sweeping results per search file (2D)
    sweepList = []
    # Index and score of best match per search file (2D)
    bestList = []

    # Compute query's MFCC
    (qRate, query) = wav.read(queryPath)
    queryMFCC = getMFCC(qRate, query)

    # List all search filenames from search pattern
    searchFileList = sorted(glob.glob(searchPatternPath))
    for searchFile in searchFileList:
        labels.append(searchFile.split("/")[-1])
        if VERBOSE:
            print("Searching '" + labels[-1] + "'...", end='')
        # Compute search file's MFCC
        (sRate, search) = wav.read(searchFile)
        searchMFCC = getMFCC(sRate, search)
        # Sweep query across search file
        sweepList.append(qbe.dtw_sweep(queryMFCC, searchMFCC, 3)) # x: time/3, y: cost
        if VERBOSE:
            print(" Done")
    
    bestList = [[0, 0] for i in range(len(sweepList))]
    # Find best match for each search file
    for i, sweep in enumerate(sweepList):
        bestList[i][0], bestList[i][1] = min(enumerate(sweep), key=operator.itemgetter(1))

    return (labels, sweepList, bestList)

def computeResults(sweepList, threshold, positiveOnly=False):
    results = []
    for i, sweep in enumerate(sweepList):
        results.append([])
        for j, score in enumerate(sweep):
            if score <= threshold:
                results[i].append([j, 1])
            else:
                if not positiveOnly:
                    results[i].append([j, 0])
    return results

def printStatistics(results, expectations):
    if VERBOSE:
        print("")
    print("Statistics")

    assert len(results) >= len(expectations)

    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    PPV = 0.0       # precision
    TPR = 0.0       # recall
    TNR = 0.0       # selectivity
    FPR = 1.0       # fall-out
    FNR = 1.0       # miss rate
    ACC = 0.0       # accuracy
    Fscore = 0.0    # f-measure
    
    TP = float(sum(np.equal(np.array(results)[:, :, 1], 1) & np.equal(np.array(expectations)[:, :, 1], 1))[0])
    TN = float(sum(np.equal(np.array(results)[:, :, 1], 0) & np.equal(np.array(expectations)[:, :, 1], 0))[0])
    FP = float(sum(np.equal(np.array(results)[:, :, 1], 1) & np.equal(np.array(expectations)[:, :, 1], 0))[0])
    FN = float(sum(np.equal(np.array(results)[:, :, 1], 0) & np.equal(np.array(expectations)[:, :, 1], 1))[0])
    print("TP:\t" + str(TP))
    print("TN:\t" + str(TN))
    print("FP:\t" + str(FP))
    print("FN:\t" + str(FN))

    if TP > 0:
        PPV = TP / (TP + FP)
        TPR = TP / (TP + FN)
        FNR = 1.0 - TPR
        Fscore = 2.0 * (PPV * TPR) / (PPV + TPR)
    if TN > 0:
        TNR = TN / (TN + FP)
        FPR = 1 - TNR
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print("Precision:\t" + str(PPV))
    print("Recall:\t\t" + str(TPR))
    print("Selectivity:\t" + str(TNR))
    print("Fall-out:\t" + str(FPR))
    print("Miss rate:\t" + str(FNR))
    print("Accuracy:\t" + str(ACC))
    print("F-Measure:\t" + str(Fscore))

def showSweeps(labels, sweepList, bestList):
    axes = plt.gca()
    #axes.set_ylim([0.2, 0.6])
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
            rgb = hexToRGB(color)
            print(get_color_escape(255, 255, 255, True), end='')
            print(get_color_escape(rgb[0], rgb[1], rgb[2]), end='')
            print(labels[i] + ": " + str(best), end='')
            print(RESET_COLOR_ESCAPE)

    axes.set_xlabel('Time', fontsize='x-large')
    axes.set_ylabel('Score', fontsize='x-large')
    axes.legend(labels)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dynamic Time Warping')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')
    parser.add_argument('query_path')
    parser.add_argument('search_pattern_path')
    args = parser.parse_args()

    VERBOSE = args.verbose
    queryPath = args.query_path
    searchPatternPath = args.search_pattern_path

    labels, sweepList, bestList = runSearch(queryPath, searchPatternPath)
    results = computeResults(sweepList, 0.4, True)

    #showSweeps(labels, sweepList, bestList)

    # XXX - temporary
    predictions = [[[0, 1]]] * len(results)

    printStatistics(results, predictions)

    plt.show()
