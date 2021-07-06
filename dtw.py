from __future__ import print_function

from scipy.fftpack import dct
from speech_dtw import qbe
from python_speech_features import fbank, lifter
from python_speech_features import delta

import utils

import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import operator

PERCENTAGE = False
VERBOSE = False

def calculate_nfft(samplerate, winlen):
    """
    source : https://github.com/jameslyons/python_speech_features/pull/76/files

    Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.
    
    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.
    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=None, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22,
         appendEnergy=True, winfunc=lambda x:np.ones((x,))):
    """
    source : https://github.com/jameslyons/python_speech_features/pull/76/files

    Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is None, which uses the calculate_nfft function to choose the smallest size that does not drop sample data.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

def getMFCC(rate, signal):
    """
    Compute MFCC, delta and delta-delta.

    :param rate: audio sample rate
    :param signal: audio signal
    :return: Cepstral Mean and Variance Normalisation
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
    
    bestList = [[0, 0] for i in range(len(sweepList))]
    # Find best match for each search file
    for i, sweep in enumerate(sweepList):
        bestList[i][0], bestList[i][1] = min(enumerate(sweep), key=operator.itemgetter(1))

    return labels, sweepList, bestList

def computeResults(sweepList, threshold, positiveOnly=False, findOnePerSweep=False):
    """
    Compare sweeping scores with the given threshold. The result is 0 if the score is 
    strictly higher than the threshold; 1 otherwise.

    :param sweepList: sweep array from search
    :param threshold: float between 0.0 and 1.0
    :param positiveOnly: boolean, True to return only positive results
    :param findOnePerSweep: boolean, True if one found query validates a sweep
    :return: array with (sweep_index, result) for each search file
    """
    # results = [sweep_index, 0 or 1]
    results = []
    for i, sweep in enumerate(sweepList):
        results.append([])
        for j, score in enumerate(sweep):
            if score <= threshold:
                if findOnePerSweep:
                    results[i].append([0, 1])
                    break
                else:
                    results[i].append([j, 1])
            else:
                if not positiveOnly and not findOnePerSweep:
                    results[i].append([j, 0])
                elif not positiveOnly and findOnePerSweep and j == len(sweep) - 1:
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
