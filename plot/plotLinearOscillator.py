## plot the linear Oscillator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.imgId import getImgId
from parameters import linOsc as lo

def l2(x):
    summe = 0
    for row in x.values:
        summe += np.nansum(row ** 2)
    return np.sqrt(summe)

def maxn(x):
    maxi = -1
    for row in x.values:
        for value in row:
            if value > maxi:
                maxi = value
    return maxi

def plotLinearOscillator( exactName, sindyName, expanName='', outName='', includeExpan=False, show=False ):
    imgId = getImgId( lo )
    tTrainEnd = int( lo.trainBatchSize * lo.tEnd )
    
    ## load data
    # exactData = pd.read_csv( f'.\\data\\linOscDopri{exactName}.csv', index_col=0 ) # load approx. data instead
    exactData = pd.read_csv( f'.\\data\\linOscExact{exactName}.csv', index_col=0 ).loc[:tTrainEnd]
    sindyData = pd.read_csv( f'.\\data\\linOscSindy{sindyName}.csv', index_col=0 ).loc[:tTrainEnd]
    expanData = pd.read_csv( f'.\\data\\linOscExpan{expanName}.csv', index_col=0 ).loc[:tTrainEnd] if includeExpan else ''

    # --------------------------------------------------------------
    # plot states (wrt time)
    # --------------------------------------------------------------
    plt.figure()

    # training area
    if lo.trainBatchSize != 1:
        plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [-2,-2,2,2], y2=0, alpha=0.05, label='Training Area')
        plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )

    if includeExpan:
        plt.plot(expanData.index, expanData['x1'], color='grey', linestyle='--', label='Fitted Series')
        plt.plot(expanData.index, expanData['x2'], color='grey', linestyle='--')
    plt.plot(exactData.index, exactData['x1'], color='r', label='$x_1$')
    plt.plot(exactData.index, exactData['x2'], color='b', label='$x_2$')
    plt.plot(sindyData.index, sindyData['x1'], color='k', linestyle=':', label='SINDy')
    plt.plot(sindyData.index, sindyData['x2'], color='k', linestyle=':')
    
    plt.xlabel('Time')
    plt.ylabel('$x_k$')

    #plt.xlim(lo.tStart, lo.tEnd)
    plt.ylim(-.2, .2)
    plt.yticks([-0.2, 0, 0.2])

    #xlim(lo.tStart, lo.tEnd)
    #plt.ylim(2, 2)
    #plt.yticks([2, 0, 2])

    #plt.title("Linear System")
    plt.legend( loc='best' )
    
    plt.savefig( f'.\\plot\\linearOscillator1{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\linearOscillator1{outName}_i{imgId}.png' )
    plt.show() if show else ''


    # --------------------------------------------------------------
    # plot limit cycle
    # --------------------------------------------------------------
    plt.figure()

    plt.plot(expanData['x1'], expanData['x2'], c='grey', linestyle='--', label='Fitted Series') if includeExpan else ''
    plt.plot(exactData['x1'], exactData['x2'], c='r', label='$x_k$')
    plt.plot(sindyData['x1'], sindyData['x2'], c='k', linestyle=':', label='SINDy')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)
    #plt.xticks([-2, 0, 2])
    #plt.yticks([-2, 0, 2])

    plt.xlim(-.2, .2)
    plt.ylim(-.2, .2)
    plt.xticks([-0.2, 0, 0.2])
    plt.yticks([-0.2, 0, 0.2])

    #plt.title("Linear System")
    plt.legend( loc='best' )

    plt.savefig( f'.\\plot\\linearOscillator2{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\linearOscillator2{outName}_i{imgId}.png' )
    plt.show() if show else ''

