## plot a pathological function
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from utils.imgId import getImgId
from parameters import pathological as pa

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

def plotPathologicalFunction( exactName, sindyName, expanName='', outName='', includeExpan=False, show=False ):
    yLim = [-5, 5]
    imgId = getImgId( pa )
    tTrainEnd = int( pa.trainBatchSize * pa.tEnd )
    
    ## load data
    # exactData = pd.read_csv( f'.\\data\\linOscDopri{exactName}.csv', index_col=0 ) # load approx. data instead
    exactData = pd.read_csv( f'.\\data\\pathologicalExact{pa.name}{exactName}.csv', index_col=0 )
    sindyData = pd.read_csv( f'.\\data\\pathologicalSindy{pa.name}{sindyName}.csv', index_col=0 )
    expanData = pd.read_csv( f'.\\data\\pathologicalExpan{pa.name}{expanName}.csv', index_col=0 ) if includeExpan else ''

    # --------------------------------------------------------------
    # plot states (wrt time)
    # --------------------------------------------------------------
    plt.figure()

    # training area
    plt.fill_between( [0,tTrainEnd, tTrainEnd,0], [yLim[0],yLim[0],yLim[1],yLim[1]], y2=0, alpha=0.05, label='Training Area')
    plt.axvline( x=tTrainEnd, c='lightgrey', lw=.5 )

    if includeExpan:
        plt.plot(expanData.index, expanData['dx'], color='grey', linestyle='--', label='Fitted Series')
    plt.plot(exactData.index, exactData['dx'], color='r', label='$f(x)$')
    plt.plot(sindyData.index, sindyData['dx'], color='k', linestyle=':', label='SINDy')
    
    plt.xlabel( '$x$' )
    plt.ylabel( '$f(x)$' )

    plt.xlim( pa.tStart, pa.tEnd )
    plt.ylim( yLim[0], yLim[1] )
    plt.yticks([ yLim[0], 0, yLim[1] ])

    #plt.title("Linear System")
    plt.legend( loc='best' )
    
    plt.savefig( f'.\\plot\\pathological1{pa.name}{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\pathological1{pa.name}{outName}_i{imgId}.png' )
    plt.show() if show else plt.close()


# --------------------------------------------------------------
# plot error (wrt tStep, polyorder or whatever)
# --------------------------------------------------------------
def plotFindings( dfName, algo, t='tStep', ylim=(0,6), norm='L2', outName='', show=False, interpolate=False ):
    df = pd.read_pickle( f'.\\data\\errordata_{dfName}.pkl' )

    assert isUnique( df['expanMethod'] )
    expanMethod = df['expanMethod'][1]

    # plot data
    label = algo if algo=='Sindy' else f'Series expansion {expanMethod}'
    if interpolate:
        x, y = df[t], df[f'{norm}Error{algo}']
        y = y.clip( upper=10 ) # y = min(y, 10)

        # interpolate with 10 times more points between x.min and x.max
        xNew = np.linspace( x.min(), x.max(), 10 * len(x) ) 
        spl = make_interp_spline( x, y, k=3 )  # type: BSpline
        ySmooth = spl( xNew )

        plt.plot( xNew, ySmooth, label=label )
    else:
        if t == 'tStep':
            plt.plot( df[t], df[f'{norm}Error{algo}'], marker='+', linestyle=" ", label=label )
        else:
            plt.plot( df[t], df[f'{norm}Error{algo}'], label=label )



    # plot exact error
    exactColor = 'grey'
    if norm == 'L2':
        if isUnique( df[f'L2ErrorExact{algo}'] ):
            plt.hlines( df[f'L2ErrorExact{algo}'][1], np.amin( df[t] ), np.amax( df[t] ), label='Analytical error', colors=exactColor )
        else:
            trendCoeffs = np.polyfit( df[t], df[f'L2ErrorExact{algo}'], 1 ) # linear trend
            if np.abs(trendCoeffs[0]) <= 1:
                plt.plot( df[t], np.polyval(trendCoeffs, df[t]), c='lightgrey', linestyle="--" )

            plt.plot( df[t], df[f'L2ErrorExact{algo}'], label='Analytical error', linestyle=' ', marker='+', c=exactColor ) # exact exact error values
            
    if t == 'tStep':
        xLabel = '$\Delta t$'
    elif t == 'polyorder':
        xLabel = 'Polynomial order'
    elif t == 'partSumOrder':
        xLabel = 'Order of the partial sum'
    else:
        xLabel = t
    plt.xlabel( xLabel )
    plt.ylabel( f'{norm} error' )

    isNotTooHigh = np.amin(df[f'{norm}Error{algo}'].values) <= ylim[1]
    plt.ylim( ylim[0], ylim[1] ) if isNotTooHigh else plt.ylim( ylim[0], 2 * np.amin(df[f'{norm}Error{algo}'].values) )

    plt.legend( loc='best' )

    xName = 'step' if t == 'tStep' else ('order' if t in ['polyorder', 'partsumorder'] else 'other')
    plt.savefig( f'.\\plot\\pathological2{dfName}{norm}{xName}{algo}{outName}.pgf' )
    plt.savefig( f'.\\plot\\pathological2{dfName}{norm}{xName}{algo}{outName}.png' )

    if show:
        plt.show()
    
    plt.close()


def isUnique( df ):
    arr = df.to_numpy()
    return (arr[0] == arr).all()