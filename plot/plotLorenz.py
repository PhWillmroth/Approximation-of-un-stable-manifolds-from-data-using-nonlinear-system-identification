## plot the Lorenz System
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.imgId import getImgId
from parameters import Lorenz as lz


def normalize( vec ):
    amin = np.amin( vec )
    amax = np.amax( vec ) + 1
    return (vec-amin) / (amax-amin) 





# --------------------------------------------------------------
# plot colorful curve
# --------------------------------------------------------------
def plotLorenzCurve( inName, outName='', cmap='jet', show=False ):
    imgId = getImgId( lz )
    tTrainEnd = int( lz.trainBatchSize * lz.tEnd )

    ## load data
    data = pd.read_csv( f'.\\data\\lorenz{inName}.csv', index_col=0 ).loc[lz.tPlotStart:lz.tPlotEnd]

    fig = plt.figure()
    ax = fig.gca( projection='3d' )

    ## plot trained area
    # Might not work on slow PCs
    ax.plot(data.loc[:tTrainEnd,'x1'], data.loc[:tTrainEnd,'x2'], data.loc[:tTrainEnd,'x3'], c='grey' )

    ## colorfull plot piecewise
    if tTrainEnd < lz.tPlotEnd:
        data = data.loc[:tTrainEnd]
        
        # precompute colors
        rkStep = normalize( data['rkStep'].values ) # number of subtimesteps used in dopri (at each timestep), normed to [0,1)
        colors = plt.get_cmap( cmap )( rkStep )

        for j in range( len(data) - 1 ):
            ax.plot( [ data.iloc[j, 0], data.iloc[j+1, 0] ],
                     [ data.iloc[j, 1], data.iloc[j+1, 1] ],
                     [ data.iloc[j, 2], data.iloc[j+1, 2] ],
                     c=colors[j] ) # the iloc wont work with other column orders!
    
    ax.set_xlim3d( -20, 20 )
    ax.set_ylim3d( -50, 50 )
    ax.set_zlim3d( 0, 50 )

    plt.locator_params(axis='x', nbins=1) # does this have any effect?
    plt.locator_params(axis='y', nbins=2)
    plt.locator_params(axis='z', nbins=3)

    ax.set_xticklabels([-20, 0, 20])
    ax.set_yticklabels([-50, 0, 50])
    ax.set_zticklabels([0, 25, 50])

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    #plt.title("Lorenz System")
    #ax.legend(loc='best')

    plt.savefig( f'.\\plot\\Lorenz1{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\Lorenz1{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()









## --------------------------------------------------------------
## plot states
## --------------------------------------------------------------
def plotLorenzState( exactName, sindyName, expanName='', outName='', includeExpan=False, show=False ):
    imgId = getImgId( lz )

    # load data
    exactData = pd.read_csv( f'.\\data\\lorenzDopri{exactName}.csv', index_col=0 ).loc[lz.tPlotStart:lz.tPlotEnd]
    sindyData = pd.read_csv( f'.\\data\\lorenzSindy{sindyName}.csv', index_col=0 ).loc[lz.tPlotStart:lz.tPlotEnd]
    expanData = pd.read_csv( f'.\\data\\lorenzExpan{expanName}.csv', index_col=0 ).loc[lz.tPlotStart:lz.tPlotEnd] if includeExpan else ''
    # --------------------------------------------------------------
    # plot t,x sindy vs exact
    # --------------------------------------------------------------
    fig = plt.figure()

    # training area
    if lz.trainBatchSize != 1:
        tTrainEnd = int( lz.trainBatchSize * lz.tEnd )
        plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [-30,-30,30,30], y2=0, alpha=0.05, label='Training Area')
        plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )

    plt.plot(expanData.index, expanData['x1'], color='grey', linestyle='--', label='Fitted Series') if includeExpan else ''
    plt.plot(exactData.index, exactData['x1'], color='r', label='$x$')
    plt.plot(sindyData.index, sindyData['x1'], color='k', linestyle=':', label='SINDy')

    plt.xlabel('Time')
    plt.ylabel('$x$')

    plt.xlim(lz.tPlotStart, lz.tPlotEnd)
    plt.ylim(-30, 30)
    plt.yticks([ -30, 0, 30 ])

    #plt.title( 'Lorenz System' )
    plt.legend(loc='best')

    plt.savefig( f'.\\plot\\Lorenz2-1{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\Lorenz2-1{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()

    # --------------------------------------------------------------
    # plot t,y sindy vs exact
    # --------------------------------------------------------------
    plt.figure()

    # training area
    if lz.trainBatchSize != 1:
        tTrainEnd = int( lz.trainBatchSize * lz.tEnd )
        plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [-30,-30,30,30], y2=0, alpha=0.05, label='Training Area')
        plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )
    
    plt.plot(expanData.index, expanData['x2'], color='grey', linestyle='--', label='Fitted Series') if includeExpan else ''
    plt.plot(exactData.index, exactData['x2'], color='r', label='$y$')
    plt.plot(sindyData.index, sindyData['x2'], color='k', linestyle=':', label='SINDy')

    plt.xlabel('Time')
    plt.ylabel('$y$')

    plt.xlim(lz.tPlotStart, lz.tPlotEnd)
    plt.ylim(-30, 30)
    plt.yticks([ -30, 0, 30 ])

    #plt.title( 'Lorenz System' )
    #plt.legend(loc='best')

    plt.savefig( f'.\\plot\\Lorenz2-2{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\Lorenz2-2{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()










# --------------------------------------------------------------
# plot l2 error
# --------------------------------------------------------------
def plotLorenzError( exactName, nameList, etaList, outName, show=False ):
    imgId = getImgId( lz )

    # errors abfangen
    names, vals = len(nameList), len(etaList)
    if names != vals:
        raise ValueError( f"The 'nameList' has {names} entries while 'etaList' has {vals}. The length must match!" )

    # load exact data
    exactData = pd.read_csv( f'.\\data\\lorenzDopri{exactName}.csv', index_col=0 ).loc[lz.tPlotStart:lz.tPlotEnd,['x1','x2','x3']]

    # Preparation
    color = ['r', 'orange', '#ffff00', 'c', 'tab:blue', 'b'] + ( ['k'] * (len(nameList)-6) )
    t = np.arange( lz.tPlotStart, lz.tPlotEnd, lz.tPlotStep )
    l2Error = np.empty( t.size ) # this will store the l2Error


    ## plot
    plt.figure()
    
    # plot l2-error
    for i in range( len(nameList) ):

        #load data
        noised = pd.read_csv( f'.\\data\\lorenz{nameList[i]}.csv', index_col=0 )
        diff = (exactData - noised.loc[lz.tPlotStart:lz.tPlotEnd,['x1','x2','x3']]).values
        
        # calculate (pointwise) l2 error
        for j in range( t.size ):
            l2Error[j] = np.linalg.norm( diff[j] )

        # plot
        plt.plot(t, l2Error, color=color[i], label=f'$\eta = {etaList[i]}$' )


    # use logarithmic scale
    plt.yscale('log')

    # draw an arrow
    plt.annotate("Increasing $\eta$", xy=(2, 10), xytext=(6, 10**(-6.5)), arrowprops=dict(arrowstyle="->"))


    # axes labels
    plt.xlabel('Time')
    plt.ylabel('Error')

    # axes limits
    plt.xlim(lz.tPlotStart, lz.tPlotEnd)
    plt.ylim( 10**(-8), 100 )

    plt.xticks([ 0, 5, 10, 15, 20 ])

    # title and legend
    #plt.title( 'Lorenz System $\ell^2$-error' )
    plt.legend()

    # save image
    plt.savefig( f'.\\plot\\Lorenz3{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\Lorenz3{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()


