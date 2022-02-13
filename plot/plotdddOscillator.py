## plot the 3D Oscillator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.imgId import getImgId
from parameters import dddOsc as do


def plotdddOscillator( exactName, sindyName, expanName='', outName='', includeExpan=False, show=False ):
    imgId = getImgId( do )
    tTrainEnd = int( do.trainBatchSize * do.tEnd )
    
    ## load data
    exactData = pd.read_csv( f'.\\data\\dddOscDopri{exactName}.csv', index_col=0 ).loc[:tTrainEnd]
    sindyData = pd.read_csv( f'.\\data\\dddOscSindy{sindyName}.csv', index_col=0 ).loc[:tTrainEnd]
    expanData = pd.read_csv( f'.\\data\\dddOscExpan{expanName}.csv', index_col=0 ).loc[:tTrainEnd] if includeExpan else ''

    # --------------------------------------------------------------
    # plot curve
    # --------------------------------------------------------------
    fig = plt.figure()
    ax = fig.gca( projection='3d' )

    ax.plot(expanData['x1'], expanData['x2'], expanData['x3'], c='grey', linestyle='--', label='Fitted Series') if includeExpan else ''
    ax.plot(exactData['x1'], exactData['x2'], exactData['x3'], c='r', label='$x_k$')
    ax.plot(sindyData['x1'], sindyData['x2'], sindyData['x3'], c='k', linestyle=':', label='SINDy')

    #ax.set_xlim3d(-2, 1.9)
    #ax.set_zlim3d(-2, 2)
    #ax.set_zlim3d(0, 1)

    plt.locator_params(axis='x', nbins=2)
    plt.locator_params(axis='y', nbins=2)
    plt.locator_params(axis='z', nbins=2)

    #ax.set_xticklabels([-2, 0, 2])
    #ax.set_yticklabels([-2, 0, 2])
    #ax.set_zticklabels([0, 0.5, 1])

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    #plt.title("3D Linear Oscillator")
    #ax.legend(loc='best')

    plt.savefig( f'.\\plot\\dddOscillator2{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\dddOscillator2{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()

    # --------------------------------------------------------------
    # plot single states
    # --------------------------------------------------------------
    plt.figure()

    # training area
    #plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [-2,-2,2,2], y2=0, alpha=0.05, label='Training Area')
    #plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )

    if includeExpan:
        plt.plot(expanData.index, expanData['x1'], color='grey', linestyle='--', label='Fitted Series')
        plt.plot(expanData.index, expanData['x2'], color='grey', linestyle='--')
    plt.plot(exactData.index, exactData['x1'], color='r', label='$x_1$')
    plt.plot(exactData.index, exactData['x2'], color='b', label='$x_2$')
    plt.plot(exactData.index, exactData['x3'], color='g', label='$x_3$')
    plt.plot(sindyData.index, sindyData['x1'], color='k', linestyle=':', label='SINDy')
    plt.plot(sindyData.index, sindyData['x2'], color='k', linestyle=':')
    plt.plot(sindyData.index, sindyData['x3'], color='k', linestyle=':')
    
    plt.xlabel('Time')
    plt.ylabel('$x_k$')

    #plt.xlim(tTrainEnd, do.tEnd)
    #plt.ylim(-2, 2)
    plt.yticks([-0.6, 0, 0.6])

    #plt.title("3D Linear Oscillator")
    plt.legend( loc='best')

    plt.savefig( f'.\\plot\\dddOscillator1{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\dddOscillator1{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()



