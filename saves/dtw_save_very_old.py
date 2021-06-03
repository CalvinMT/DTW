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

def showResults(labels, sweepList):
    axes = plt.gca()
    axes.set_ylim([0.2, 0.6])
    scoreList = [[0, 0] for i in range(len(sweepList))] # 0: index; 1: minimum
    for i, sweep in enumerate(sweepList):
        color = next(axes._get_lines.prop_cycler)['color']
        plt.plot(sweep, ':', color=color, zorder=i)
        # plot minimum point
        scoreList[i][0], scoreList[i][1] = min(enumerate(sweep), key=operator.itemgetter(1))

    axes.set_prop_cycle(None)
    for i, score in enumerate(scoreList):
        color = next(axes._get_lines.prop_cycler)['color']
        plt.plot(score[0], score[1], 'o', color=color, zorder=i+len(sweepList)-1)
        if VERBOSE:
            rgb = hexToRGB(color)
            print(get_color_escape(255, 255, 255, True), end='')
            print(get_color_escape(rgb[0], rgb[1], rgb[2]), end='')
            print(labels[i] + ": " + str(score), end='')
            print(RESET_COLOR_ESCAPE)

    axes.set_xlabel('Time', fontsize='x-large')
    axes.set_ylabel('Score', fontsize='x-large')
    axes.legend(labels)
    plt.show()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dynamic Time Warping')
    #parser.add_argument('-d', '--debug', action='store_true', help='Enable debug display')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')
    parser.add_argument('query_path')
    parser.add_argument('search_pattern_path')
    args = parser.parse_args()

    VERBOSE = args.verbose
    queryPath = args.query_path
    searchPatternPath = args.search_pattern_path

    # Search filenames (1D)
    labels = []
    # Sweeping results per search file (2D)
    sweepList = []

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
        sweepList.append(qbe.dtw_sweep(queryMFCC, searchMFCC, 1)) # x: time/3, y: cost
        if VERBOSE:
            print(" Done")

    if VERBOSE:
        print("")
        print("Best match (position, score):")
    showResults(labels, sweepList)
