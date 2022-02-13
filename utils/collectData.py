import pandas as pd
from matplotlib import pyplot as plt
import os

def clearOldFindings(clazz, version):
    PATH = getDataPath(clazz, version)
    if os.path.isfile( PATH ):
        consent = input( '??? Delete pkl (y/n):  ' )
        if consent == 'y':
            os.remove( PATH )
        print('ok\n')


def getDataPath(clazz, version):
    clazzName = clazz.__name__
    return f'./data/errordata_{clazzName}v{version}.pkl' if not clazzName == 'pathological' else f'./data/errordata_{clazzName}v{version}.pkl'

def saveFindings( clazz, version, **kwargs):
    PATH = getDataPath(clazz, version)

    # create df from input data
    classAttrs = dict(vars( clazz ).items()) # get all attributes of 'clazz'
    classAttrs.pop('__module__') # delete tuple
    classAttrs.pop('__doc__')
    classAttrs.update( kwargs ) # merge input arguments with class attributes
    if 'functionDict' in classAttrs: classAttrs['functionDict'] = 'non-monomial' # e.g. legendre
    if 'dOmg1' in classAttrs: classAttrs['dOmg1'] = str(classAttrs['dOmg1'])
    if 'dOmg2' in classAttrs: classAttrs['dOmg2'] = str(classAttrs['dOmg2'])
    if 'x0' in classAttrs: classAttrs['x0'] = str(classAttrs['x0']) # otherwise DF will be confused with dimensions (so no arrays allowed)
    if 'xi' in classAttrs: classAttrs['xi'] = str(classAttrs['xi']) # otherwise DF will be confused with dimensions (so no arrays allowed)

    newDf = pd.DataFrame( classAttrs, index=[pd.Timestamp('today')] )

    # save dfs
    if os.path.isfile( PATH ):
        # read and concatenate existing pkl
        oldDf = pd.read_pickle( PATH )
        pd.concat((oldDf, newDf)).to_pickle( PATH )
    else:
        print('Create new pkl for', clazz.__name__)
        newDf.to_pickle( PATH )

    
    

def plotFindings( dfName, att1, att2 ):
    df = pd.read_pickle( f'data\\errordata_{dfName}.pkl' )
    plt.plot(df[att1], df[att2])
    plt.yscale('log')
    plt.title(dfName)
    plt.show()

def plotMultipleFindings( dfName ):
    df = pd.read_pickle( f'data\\errordata_{dfName}.pkl' )

    t = df['phiScale']

    plt.plot(t, df['linOsc'], label='Linear oscillator')
    plt.plot(t, df['cubOsc'], label='Cubic oscillator')
    plt.plot(t, df['dddOsc'], label='3D linear oscillator')
    plt.plot(t, df['lorenz'], label='Lorenz system')
    plt.plot(t, df['doubPend'], label='Double pendulum')

    # Vertical line
    plt.vlines(x=1, ymin=-0.1, ymax=1.1, colors='grey', linestyles="--")

    plt.xlabel('$m / \\varphi$')
    plt.ylabel('$\mathrm{rk}(\\Theta(X)) / \\varphi$')

    plt.legend(loc='lower right')


    plt.savefig( f'.\\plot\\errordata_{dfName}.png' )
    plt.savefig( f'.\\plot\\errordata_{dfName}.pgf' )

    plt.show()


def plotMultipleFindings2( dfName, M, norm ):
    df = pd.read_pickle( f'data\\errordata_{dfName}.pkl' )

    df1 = df.where(df['M'] == M)
    #df2 = df.where(df['M'] == 60)
    #df3 = df.where(df['M'] == 30)
    #df4 = df.where(df['M'] == 2)

    plt.plot(df1['K'], df1['L2Error'] / norm, label=f'${M}$ samples', c='darkgreen')
    #plt.plot(df2['m'], df2['L2Error'])
    #plt.plot(df3['m'], df3['L2Error'])
    #plt.plot(df4['m'], df4['L2Error'])

    plt.xlabel('Number of bursts $K$')
    plt.ylabel('rel. $L^2$-error')

    plt.yscale('log')

    plt.legend()
    plt.savefig( f'.\\plot\\errordata_{dfName}.png' )
    plt.savefig( f'.\\plot\\errordata_{dfName}.pgf' )

    plt.show()

def plotMultipleFindings3( dfName, norm ):
    df = pd.read_pickle( f'data\\errordata_{dfName}.pkl' )
    df = pd.read_pickle( f'data\\errordata_{dfName}.pkl' )

    plt.plot(df['m'], df['L2Error'] / norm, label=f'Cubic oscillator initialised at the Padua points', c='darkgreen')

    plt.xlabel('Number of samples $m$')
    plt.ylabel('rel. $L^2$-error')

    plt.yscale('log')

    plt.legend()
    plt.savefig( f'.\\plot\\errordata_{dfName}.png' )
    plt.savefig( f'.\\plot\\errordata_{dfName}.pgf' )

    plt.show()

def plotMultipleFindings4( dfName1, dfName2, norm ):
    df1 = pd.read_pickle( f'data\\errordata_{dfName1}.pkl' )
    df2 = pd.read_pickle( f'data\\errordata_{dfName2}.pkl' )

    plt.plot(df1['m'], df1['L2Error'] / norm, label=f'Initialised at the Padua points', c='darkgreen')
    plt.plot(df2['m'], df2['L2Error'] / norm, label=f'Randomly initialised', c='grey')

    plt.xlabel('Number of samples $m$')
    plt.ylabel('rel. $L^2$-error')

    plt.yscale('log')

    plt.legend()
    plt.savefig( f'.\\plot\\errordata_{dfName1}_{dfName2}.png' )
    plt.savefig( f'.\\plot\\errordata_{dfName1}_{dfName2}.pgf' )

    plt.show()

norm = 1
def plotTwoFindings(dfName, x, y1, y2, label1=None, label2= None, title=None, save=True):
    df = pd.read_pickle( f'data\\errordata_{dfName}.pkl' )
    plt.plot(df[x], df[y1] / norm, label=label1, c="green")
    plt.plot(df[x], df[y2] / norm, label=label2, c="blue", linestyle="--")

    plt.title(title if title else dfName)
    if label1 or label2: plt.legend()

    if save:
        plt.savefig( f'.\\plot\\errordata_{dfName}.png' )
        plt.savefig( f'.\\plot\\errordata_{dfName}.pgf' )
    plt.show()