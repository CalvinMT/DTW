from dtw import computeResults

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

VERBOSE = False

def printStatistics(results, expectations):
    """
    Compare results with expectations and print out some statistics.

    :param results: array of obtained results
    :param expectations: array of expected results
    :return: None
    """
    if VERBOSE:
        print("")
    print("Statistics")

    TP, TN, FP, FN = getConfusionMatrix(results, expectations)
    print("TP:\t" + str(TP))
    print("TN:\t" + str(TN))
    print("FP:\t" + str(FP))
    print("FN:\t" + str(FN))

    # recall, fall-out
    TPR, FPR = getTPRAndFPR(TP, TN, FP, FN)
    print("Recall:\t\t" + str(TPR))
    print("Fall-out:\t" + str(FPR))

    PPV = 0.0       # precision
    TNR = 0.0       # selectivity
    FNR = 1.0       # miss rate
    ACC = 0.0       # accuracy
    Fscore = 0.0    # f-measure

    if TP > 0:
        PPV = TP / (TP + FP)
        FNR = 1.0 - TPR
        Fscore = 2.0 * (PPV * TPR) / (PPV + TPR)
    if TN > 0:
        TNR = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print("Precision:\t" + str(PPV))
    print("Selectivity:\t" + str(TNR))
    print("Miss rate:\t" + str(FNR))
    print("Accuracy:\t" + str(ACC))
    print("F-Measure:\t" + str(Fscore))

def getConfusionMatrix(results, expectations):
    """
    Compare results with expectations and return the number of true positive (TP),
    true negative (TN), false positive (FP), false negative (FN) respectively.
    The results array and the expectations array don't have to be of the same length,
    though the results array's length have to be equal or higher than the expectations
    array's length.

    :param results: array of obtained results
    :param expectations: array of expected results
    :return: tuple of four floats
    """
    assert len(results) >= len(expectations)
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for i in range(len(results)):
        r = 0
        e = 0
        while r < len(results[i]) or (i < len(expectations) and e < len(expectations[i])):
            if i >= len(expectations) or e >= len(expectations[i]) or (r < len(results[i]) and results[i][r][0] < expectations[i][e][0]):
                if results[i][r][1]:
                    FP += 1.0
                else:
                    TN += 1.0
            elif r >= len(results[i]) or (e < len(expectations[i]) and results[i][r][0] > expectations[i][e][0]):
                FN += 1.0
                r -= 1
                e += 1
            else:
                if results[i][r][1] == expectations[i][e][1]:
                    if results[i][r][1]:
                        TP += 1.0
                    else:
                        TN += 1.0
                else:
                    if results[i][r][1]:
                        FP += 1.0
                    else:
                        FN += 1.0
                e += 1
            r += 1
    return TP, TN, FP, FN

def getTPRAndFPR(TP, TN, FP, FN):
    """
    Compute and return the true positive rate (TPR) and the false positive rate (FPR)
    from the number of true positive (TP), true negative (TN), false positive (FP),
    false negative (FN) respectively.

    :param TP: float
    :param TN: float
    :param FP: float
    :param FN: float
    :return: tuple of two floats
    """
    TPR = 0.0       # recall
    FPR = 0.0       # fall-out
    if TP > 0.0:
        TPR = TP / (TP + FN)
    if TN > 0.0:
        FPR = TN / (TN + FP)
    return TPR, FPR

def computeROCCurve(sweepList, expectations, nbThresholds=100, positiveOnly=False, oneWord=False):
    """
    TODO

    :return: AUC, ROC curve pivots (x, y, thresholds)
    """
    thresholdList = list(np.array(list(range(0, nbThresholds + 1, 1))) / float(nbThresholds))
    rocPoints = []

    for threshold in thresholdList:
        results = computeResults(sweepList, threshold, positiveOnly, oneWord)
        TP, TN, FP, FN = getConfusionMatrix(results, expectations)
        TPR, FPR = getTPRAndFPR(TP, TN, FP, FN)
        rocPoints.append([TPR, FPR])
    
    pivot = pd.DataFrame(rocPoints, columns=["x", "y"])
    pivot["thresholds"] = thresholdList

    AUC = abs(np.trapz(pivot.x, pivot.y))

    return AUC, pivot

def showROCCurve(AUC, pivot, figure=None, plotpos=[1, 1, 1], title="Receiver Operating Characteristic"):
    """
    TODO

    :return: None
    """
    if figure == None:
        figure = plt.figure()
    ax = figure.add_subplot(plotpos[0], plotpos[1], plotpos[2])
    ax.set_title(title + "\n(AUC=" + str(round(AUC, 4)) + ")")
    ax.plot(np.array(pivot.y), np.array(pivot.x))
    ax.plot([1, 0],'r--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.invert_xaxis()
