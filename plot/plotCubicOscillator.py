## plot the cubic Oscillator
from utils.exactError import exactErrorSindy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from utils.imgId import getImgId
from parameters import cubOsc as cu


def plotCubicOscillator( exactName, sindyName, expanName='', outName='', includeExpan=False, show=False ):
    imgId = getImgId( cu )
    tTrainEnd = int( cu.trainBatchSize * cu.tEnd )
    
    ## load data
    exactData = pd.read_csv( f'.\\data\\cubOscDopri{exactName}.csv', index_col=0 ).loc[:tTrainEnd]
    sindyData = pd.read_csv( f'.\\data\\cubOscSindy{sindyName}.csv', index_col=0 ).loc[:tTrainEnd]
    expanData = pd.read_csv( f'.\\data\\cubOscExpan{expanName}.csv', index_col=0 ).loc[:tTrainEnd] if includeExpan else '' 

    # --------------------------------------------------------------
    # plot states (wrt time)
    # --------------------------------------------------------------
    plt.figure()

    # training area
    plt.fill_between( [-2,tTrainEnd, tTrainEnd,-2], [-2,-2,2,2], y2=0, alpha=0.05, label='Training Area')
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

    #plt.xlim(tTrainEnd, cu.tEnd)
    plt.ylim(-2, 2)
    plt.xlim(-2, 52)
    plt.yticks([-2, 0, 2])

    #plt.title("Cubic Nonlinearity")
    plt.legend(loc='best')

    plt.savefig( f'.\\plot\\cubicOscillator1{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\cubicOscillator1{outName}_i{imgId}.png' )
    plt.show() if show else ''



    # --------------------------------------------------------------
    # plot limit cycle
    # --------------------------------------------------------------
    fig = plt.figure()

    plt.plot(expanData['x1'], expanData['x2'], c='grey', linestyle='--', label='Fitted Series') if includeExpan else ''
    plt.plot(exactData['x1'], exactData['x2'], c='r', label='$x_k$')
    plt.plot(sindyData['x1'], sindyData['x2'], c='k', linestyle=':', label='SINDy')

    #plt.scatter(exactData.loc[tTrainEnd]['x1'], exactData.loc[tTrainEnd]['x2'], s=75, c='darkgrey',
    #            marker='x', label='Training Barrier') # training barrier
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])

    #plt.title("Cubic Nonlinearity")
    #plt.legend(['$x_k$', 'model'], loc='best')

    plt.savefig( f'.\\plot\\cubicOscillator2{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\cubicOscillator2{outName}_i{imgId}.png' )
    plt.show() if show else ''


# --------------------------------------------------------------
# plot error (wrt tStep, polyorder or whatever)
# --------------------------------------------------------------
def plotCuFindings( dfName, algo, t='tStep', ylim=None, norm='L2', outName='', show=False, interpolate=False, accumulate=False, log=False, startInd=0, label=None, legend=False ):
    df = pd.read_pickle( f'.\\data\\errordata_{dfName}.pkl' ).iloc[startInd:]

    assert isUnique( df['expanMethod'] )
    expanMethod = df['expanMethod'][1]

    # sort
    df = df.sort_values(by=t)

    # preprocess data
    values = df[f'{norm}Error{algo}']
    if accumulate:
        for i in range(1, values.size-1):
            values.iloc[i] = 0.5 * (0.5*values.iloc[i-1] +  values.iloc[i] +  0.5*values.iloc[i+1])

    ## PLOT
    label = label if label else (algo if algo=='Sindy' else f'Series expansion {expanMethod}')
    if interpolate:
        x, y = df[t], values
        y = y.clip( upper=10 ) # y = min(y, 10)

        # interpolate with 10 times more points between x.min and x.max
        xNew = np.linspace( x.min(), x.max(), 10 * len(x) ) 
        spl = make_interp_spline( x, y, k=3 )  # type: BSpline
        ySmooth = spl( xNew )

        plt.plot( xNew, ySmooth, label=label )
    else:
        # PLOT
        if t == 'tStep':
            plt.plot( df[t], values, marker='+', linestyle=" ", label=label )
        else:
            plt.plot( df[t], values/18972579.099136498, label=label, c='darkgreen' )

    if t == 'tStep':
        xLabel = '$\Delta t$'
    elif t == 'polyorder':
        xLabel = 'Polynomial order'
    elif t == 'partSumOrder':
        xLabel = 'Order of the partial sum'
    elif t == 'eta':
        xLabel = 'Standard deviation $\sigma$'
    elif t == 'lmbd':
        xLabel = 'Sparsification threshold $\lambda$'
        # invert x axis:
        # plt.axis([np.amax(df[t].values), \
        #             np.amin(df[t].values), \
        #             np.amin(df[f'{norm}Error{algo}'].values), \
        #             np.amax(df[f'{norm}Error{algo}'].values)] )
    else:
        xLabel = t

    if norm == 'relL2':
        yLabel = 'rel. $L^2$-error'
    elif norm == 'L2':
        yLabel = 'rel. $L^2$-error'
    else:
        yLabel = f'{norm} error'

    plt.xlabel( xLabel )
    plt.ylabel( yLabel )

    if ylim:
        isNotTooHigh = np.amin(df[f'{norm}Error{algo}'].values) <= ylim[1]
        plt.ylim( ylim[0], ylim[1] ) if isNotTooHigh else plt.ylim( ylim[0], 2 * np.amin(df[f'{norm}Error{algo}'].values) )

    plt.legend( loc='best' ) if legend else ''
    plt.yscale('log') if log else ''

    xName = 'step' if t == 'tStep' else ('order' if t in ['polyorder', 'partsumorder'] else 'other')
    plt.savefig( f'.\\plot\\cubOsc3{dfName}{norm}{xName}{algo}{"ACC" if accumulate else ""}{"Log" if log else ""}{outName}.pgf' )
    plt.savefig( f'.\\plot\\cubOsc3{dfName}{norm}{xName}{algo}{"ACC" if accumulate else ""}{"Log" if log else ""}{outName}.png' )

    if show:
        plt.show()
    
    plt.close()

def isUnique( df ):
    arr = df.to_numpy()
    return (arr[0] == arr).all()

def plotCuDerivative( exactName, outName='', show=False ):
    imgId = getImgId( cu )
    
    ## load data
    exactData = pd.read_csv( f'.\\data\\cubOscDopri{exactName}.csv', index_col=0 ).loc[:(cu.tEnd/2)]

    ## add noise
    exactData['dx1Noi'], exactData['dx2Noi'] = exactData['dx1'], exactData['dx2']
    exactData[['dx1Noi', 'dx2Noi']] += cu.eta * cu.randn( exactData[['dx1', 'dx2']].shape )

    # --------------------------------------------------------------
    # plot derivative (wrt time)
    # --------------------------------------------------------------
    plt.figure()
    plt.plot(exactData.index, exactData['dx1Noi'], color='grey', label='$\dot x_k$ noised')
    plt.plot(exactData.index, exactData['dx2Noi'], color='grey')
    plt.plot(exactData.index, exactData['dx1'], color='r', label='$\dot x_1$')
    plt.plot(exactData.index, exactData['dx2'], color='b', label='$\dot x_2$')
    
    plt.xlabel('Time')
    plt.ylabel('$\dot x_k$')

    plt.xlim(cu.tStart, cu.tEnd/2)
    #plt.ylim(-2, 2)
    plt.yticks([-15, 0, 10])

    #plt.title("Cubic oscillator")
    #plt.legend()

    plt.savefig( f'.\\plot\\cubicOscillator4{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\cubicOscillator4{outName}_i{imgId}.png' )
    plt.show() if show else ''


def plotProb( dfName, show=True ):
    df = pd.read_pickle( f'.\\data\\errordata_{dfName}.pkl' ).iloc[1:]

    df[['c1', 'c2']] = np.log(1 / (1 - df[['c1', 'c2']]))

    for i in range(len(df)):
        row = df.iloc[i]
        plt.scatter( row['K'], row['c1'], marker='x', c='blue')
        plt.scatter( row['K'], row['c2'], marker='x', c='green')
        #plt.scatter( row['K'], row['c3'], marker='x', c='red')
        #plt.scatter( row['K'], row['c4'], marker='x', c='orange')
    plt.scatter( df.iloc[-1]['K'], df.iloc[-1]['c1'], marker='x', c='blue', label='$f_1$') # for the legend only
    plt.scatter( df.iloc[-1]['K'], df.iloc[-1]['c2'], marker='x', c='green', label='$f_2$') # for the legend only

    # boundary line
    m = np.amin(df['c2'] / df['K'])
    print('c-star', m)
    plt.plot([0,105], [0,(105*m)-0.0125], c='grey', label='Approximated boundary', linestyle='--')

    plt.xlabel('Number of bursts $K$')
    plt.ylabel("$\log\\big( \\varepsilon^{-1}\\big)$")   
    #plt.yticks([0, 0.1, 0.2, 0.3]) 
    plt.legend(loc='upper left')
    
    plt.savefig( f'.\\plot\\cubicOscillatorErrordata_{dfName}.pgf' )
    plt.savefig( f'.\\plot\\cubicOscillatorErrordata_{dfName}.png' )
    plt.show() if show else ''


def plotInitialisation(normalPath, burstPathList, mMax, outName="", show=False, plotNormal=True):

    if plotNormal:
        df1 = pd.read_csv( f'.\\data\\cubOscDopri{normalPath}.csv', index_col=0 ).iloc[:mMax]
        #plt.plot(df1['x1'], df1['x2'], c="grey", alpha=.3)
        plt.scatter(df1['x1'], df1['x2'], c=((df1['dx1'] ** 2) + (df1['dx2'] ** 2)), cmap="bwr", label="Single initialisation")

    for path in burstPathList:
        df = pd.read_csv( f'.\\data\\cubOscDopri{path}.csv', index_col=0 ).iloc[:int(mMax/len(burstPathList))]
        #plt.plot(df['x1'], df['x2'], c="grey", alpha=.3)
        plt.scatter(df['x1'], df['x2'], c=((df['dx1'] ** 2) + (df['dx2'] ** 2)), cmap="bwr", marker="x")
    
    if burstPathList:
        plt.scatter(3,3, c="red", marker="x", label="Multiple random initialisations") # for legend


    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.xlim(-2,2)
    plt.ylim(-2, 2)

    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])

    plt.legend( loc='best' )

    plt.savefig( f'.\\plot\\cubicOscillator3{outName}.pgf' )
    plt.savefig( f'.\\plot\\cubicOscillator3{outName}.png' )
    if show: plt.show()
    plt.close()