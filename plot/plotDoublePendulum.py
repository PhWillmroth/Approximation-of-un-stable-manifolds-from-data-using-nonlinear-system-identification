## plot the double Pendulum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from utils.imgId import getImgId
from parameters import doubPend as dp


def normalise( vec ):
    amin = np.amin( vec )
    amax = np.amax( vec ) + 1
    return (vec-amin) / (amax-amin) 





# --------------------------------------------------------------
# plot colorful curve
# --------------------------------------------------------------
def plotDoubPendCurve( exactName, sindyName, expanName='', outName='', cmap='jet', includeExpan=False, show=False ):
    imgId = getImgId( dp )

    # load, cut and normalise data
    exactData = pd.read_csv( f'.\\data\\doubPendDopri{exactName}.csv', index_col=0, float_precision='round_trip').loc[:dp.tPlotEnd] # pandas rounding. see https://stackoverflow.com/questions/47368296/pandas-read-csv-file-with-float-values-results-in-weird-rounding-and-decimal-dig
    sindyData = pd.read_csv( f'.\\data\\doubPendSindy{sindyName}.csv', index_col=0 ).loc[:dp.tPlotEnd]
    expanData = pd.read_csv( f'.\\data\\doubPendExpan{expanName}.csv', index_col=0 ).loc[:dp.tPlotEnd] if includeExpan else ''
    exactData['rkStep'] = normalise( exactData['rkStep'].values ) # normalise rkStep to [0,1)
    
    # go from phase space to state space [x1,y1,x2,y2]
    height = dp.l1 + dp.l2
    exactStates = pd.DataFrame([ dp.l1 * np.sin( exactData['phi1'] ) + dp.l2 * np.sin( exactData['phi2'] ),
                                height - dp.l2 * np.cos( exactData['phi1'] ) - dp.l2 * np.cos( exactData['phi2'] ) ],
                                index=['x1','x2'], columns=exactData.index ).T
    sindyStates = pd.DataFrame([ dp.l1 * np.sin( sindyData['phi1'] ) + dp.l2 * np.sin( sindyData['phi2'] ),
                                height - dp.l2 * np.cos( sindyData['phi1'] ) - dp.l2 * np.cos( sindyData['phi2'] )  ],
                                index=['x1','x2'], columns=sindyData.index ).T
    expanStates = pd.DataFrame([ dp.l1 * np.sin( expanData['phi1'] ) + dp.l2 * np.sin( expanData['phi2'] ),
                                height - dp.l2 * np.cos( expanData['phi1'] ) - dp.l2 * np.cos( expanData['phi2'] )  ],
                                index=['x1','x2'], columns=expanData.index ).T if includeExpan else ''

    exactData = pd.concat([exactData, exactStates], axis=1)
    sindyData = pd.concat([sindyData, sindyStates], axis=1)
    expanData = pd.concat([expanData, expanStates], axis=1) if includeExpan else ''

    ## plot
    plt.figure()

    # plot pendulum scetch
    plt.plot( [0,0], [0,2], c='#dddddd' )
    plt.scatter( 0, 1, c='#dddddd', s=13, marker='o')
    plt.scatter( 0, 0, c='#dddddd', marker='o')

    # plot expan data
    plt.plot( expanData['x1'], expanData['x2'], c='grey', linestyle='--', label='Fitted Series' ) if includeExpan else ''

    # piecewise colorful exact data plot
    for j in range( len(exactData) - 1):
        plt.plot( [ exactData['x1'].iloc[j], exactData['x1'].iloc[j+1] ],
                  [ exactData['x2'].iloc[j], exactData['x2'].iloc[j+1] ],
                  c = plt.get_cmap( cmap )(exactData['rkStep'].iloc[j]) )

    # plot sindy data
    plt.plot( sindyData['x1'], sindyData['x2'], c='k', linestyle=':' )

 
    plt.xticks( [-1, 0, 1] )
    plt.yticks( [0, 1, 2] )
    
    #plt.xlim( -2, 2 )
    #plt.ylim( -0.2, 2.8 )
    
    plt.xlabel( '$x_1$' )
    plt.ylabel( '$x_2$' )

    # legend and title
    #plt.title( 'Double Pendulum' )
    legend_elements = [ Line2D([0], [0], color=plt.get_cmap( cmap )(.5), label='$x_k$')#Patch(facecolor='g', label='$x_k$'),
                        ,Line2D([0], [0], color='k', linestyle=':', label='SINDy')] + ( [Line2D([0], [0], color='grey', linestyle='--', label='Fitted Series'),] if includeExpan else [])
    plt.legend( handles=legend_elements, loc='best' )


    plt.savefig( f'.\\plot\\doublePendulum1{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\doublePendulum1{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()









## --------------------------------------------------------------
## plot states individually
## --------------------------------------------------------------
def plotDoubPendState( exactName, sindyName, expanName='', outName='', includeExpan=False, show=False ):
    imgId = getImgId( dp )
    tTrainEnd = int( dp.trainBatchSize * dp.tEnd )

    # load and cut data
    exactData = pd.read_csv( f'.\\data\\doubPendDopri{exactName}.csv', index_col=0 ).loc[tTrainEnd:dp.tPlotEnd]
    sindyData = pd.read_csv( f'.\\data\\doubPendSindy{sindyName}.csv', index_col=0 ).loc[tTrainEnd:dp.tPlotEnd]
    expanData = pd.read_csv( f'.\\data\\doubPendExpan{expanName}.csv', index_col=0 ).loc[tTrainEnd:dp.tPlotEnd] if includeExpan else ''

    # plot delimiters
    xlim0, xlim1 = tTrainEnd, dp.tPlotEnd
    ylim = 2.5

    # --------------------------------------------------------------
    # model vs exact, plot t,x 
    # --------------------------------------------------------------
    plt.figure()

    # training area
    if dp.trainBatchSize != 1:
        plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [-2*np.pi, -2*np.pi, 2*np.pi, 2*np.pi], y2=0, alpha=0.05, label='Training Area')
        plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )

    plt.plot(expanData.index, expanData['phi1'], color='grey', linestyle='--', label='Fitted Series') if includeExpan else ''
    plt.plot(exactData.index, exactData['phi1'], color='r', label='$\\varphi_1$')
    plt.plot(sindyData.index, sindyData['phi1'], color='k', linestyle=':', label='SINDy')

    plt.xlabel('Time')
    plt.ylabel('$\\varphi_1$')

    plt.xlim( xlim0, xlim1 )
    plt.ylim( -ylim, ylim )
    plt.yticks([ -ylim, 0, ylim ])

    #plt.title( 'Double Pendulum' )
    plt.legend( loc='best')

    plt.savefig( f'.\\plot\\doublePendulum2-1{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\doublePendulum2-1{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()


    # --------------------------------------------------------------
    # model vs exact, plot t,y 
    # --------------------------------------------------------------
    plt.figure()

    # training area
    #plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [-2*np.pi, -2*np.pi, 2*np.pi, 2*np.pi], y2=0, alpha=0.05, label='Training Area')
    #plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )

    plt.plot(expanData.index, expanData['phi2'], color='grey', linestyle='--', label='Fitted Series') if includeExpan else ''
    plt.plot(exactData.index, exactData['phi2'], color='r', label='$\\varphi_2$')
    plt.plot(sindyData.index, sindyData['phi2'], color='k', linestyle=':', label='SINDy')

    plt.xlabel('Time')
    plt.ylabel('$\\varphi_2$')

    plt.xlim(xlim0, xlim1)
    plt.ylim( -ylim, ylim )
    plt.yticks([ -ylim, 0, ylim ])

    #plt.title( 'Double Pendulum' )
    #plt.legend( loc='best')

    plt.savefig( f'.\\plot\\doublePendulum2-2{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\doublePendulum2-2{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()






## --------------------------------------------------------------
## plot states individually: compare two SINDy outcomes
## --------------------------------------------------------------
def plotDoubPendStateDict( exactName, sindyName, sindyName2, outName='', show=False ):
    imgId = getImgId( dp )
    tTrainEnd = int( dp.trainBatchSize * dp.tEnd )

    # load and cut data
    exactData = pd.read_csv( f'.\\data\\doubPendDopri{exactName}.csv', index_col=0 ).loc[:dp.tPlotEnd]
    sindyData = pd.read_csv( f'.\\data\\doubPendSindy{sindyName}.csv', index_col=0 ).loc[:dp.tPlotEnd]
    sindy2Data = pd.read_csv( f'.\\data\\doubPendSindy{sindyName2}.csv', index_col=0 ).loc[:dp.tPlotEnd]

    # plot delimiters
    xlim0, xlim1 = dp.tPlotStart, dp.tPlotEnd
    ylim = 2.5

    # --------------------------------------------------------------
    # model vs exact, plot t,x 
    # --------------------------------------------------------------
    plt.figure()

    # training area
    plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [-2*np.pi, -2*np.pi, 2*np.pi, 2*np.pi], y2=0, alpha=0.05, label='Training Area')
    plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )

    plt.plot(sindy2Data.index, sindy2Data['phi1'], color='grey', linestyle='--', label='Polynomial dictionary')
    plt.plot(exactData.index, exactData['phi1'], color='r', label='$\\varphi_1$')
    plt.plot(sindyData.index, sindyData['phi1'], color='k', linestyle=':', label='Artificial dictionary')

    plt.xlabel('Time')
    plt.ylabel('$\\varphi_1$')

    plt.xlim( xlim0, xlim1 )
    plt.ylim( -ylim, ylim )
    plt.yticks([ -ylim, 0, ylim ])

    #plt.title( 'Double Pendulum' )
    plt.legend( loc='best')

    plt.savefig( f'.\\plot\\doublePendulum2-3{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\doublePendulum2-3{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()


    # --------------------------------------------------------------
    # model vs exact, plot t,y 
    # --------------------------------------------------------------
    plt.figure()

    # training area
    plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [-2*np.pi, -2*np.pi, 2*np.pi, 2*np.pi], y2=0, alpha=0.05, label='Training Area')
    plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )

    plt.plot(sindy2Data.index, sindy2Data['phi2'], color='grey', linestyle='--', label='Polynomial dictionary')
    plt.plot(exactData.index, exactData['phi2'], color='r', label='$\\varphi_2$')
    plt.plot(sindyData.index, sindyData['phi2'], color='k', linestyle=':', label='Artificial dictionary')

    plt.xlabel('Time')
    plt.ylabel('$\\varphi_2$')

    plt.xlim(xlim0, xlim1)
    plt.ylim( -ylim, ylim )
    plt.yticks([ -ylim, 0, ylim ])

    #plt.title( 'Double Pendulum' )
    #plt.legend( loc='best')

    plt.savefig( f'.\\plot\\doublePendulum2-4{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\doublePendulum2-4{outName}_i{imgId}.png' )
    plt.show() if show else ''
    plt.close()










# --------------------------------------------------------------
# plot l2 error
# --------------------------------------------------------------
def plotDoubPendError( exactName, nameList, etaNameList, outName='', show=False ):
    imgId = getImgId( dp )

    # errors abfangen
    names, vals = len(nameList), len(etaNameList)
    assert names == vals, f"The 'nameList' has {names} entries while 'etaList' has {vals}. The length must match."

    # load exact data
    exactData = pd.read_csv( f'.\\data\\doubPendDopri{exactName}.csv', index_col=0 ).loc[:dp.tPlotEnd, ['phi1','phi2']]

    # Preparation
    color = ['r', 'orange', '#ffff00', 'c', 'tab:blue', 'b'] + ( ['k'] * (len(nameList)-6) )
    #t = np.arange( dp.tPlotStart, dp.tPlotEnd, dp.tPlotStep )
    l2Error = np.empty( len(exactData) ) # this will store the l2Error


    ## plot
    plt.figure()
    
    # plot l2-error
    for i in range( len(nameList) ):

        #load data
        noised = pd.read_csv( f'.\\data\\DoubPend{nameList[i]}.csv', index_col=0 )
        diff = (exactData - noised.loc[:dp.tPlotEnd, ['phi1','phi2']]).values
        
        # calculate (pointwise) l2 error
        for j in range( len(l2Error) ):
            l2Error[j] = np.linalg.norm( diff[j] )

        # plot
        plt.plot(exactData.index, l2Error, color=color[i], label=f'$\eta = {etaNameList[i]}$' )


    # use logarithmic scale
    plt.yscale('log')

    # draw an arrow
    plt.annotate("Increasing $\eta$", xy=(4, 10), xytext=(4.5, 10**(-6.5)), arrowprops=dict(arrowstyle="->"))


    # axes labels
    plt.xlabel('Time')
    plt.ylabel('Error')

    # axes limits
    plt.xlim(dp.tPlotStart, dp.tPlotEnd)
    plt.ylim( 10**(-8), 100 )

    #plt.xticks([ 0, 5, 10, 15, 20 ])

    # title and legend
    #plt.title( 'Double Pendulum $\ell^2$-error' )
    plt.legend()

    # save image
    plt.savefig( f'.\\plot\\doublePendulum3{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\doublePendulum3{outName}_i{imgId}.png' )
    plt.show() if show else ''



