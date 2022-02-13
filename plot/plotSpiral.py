## plot the spiral
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.imgId import getImgId
from parameters import spiral as sp

def l2error(exactName, tossiName):
    ## load data
    exact = pd.read_csv(f'.\\data\\spiralExact{exactName}.csv', index_col=0)
    tossi = pd.read_csv(f'.\\data\\spiralTossi{tossiName}.csv', index_col=0)

    print( sp.method, np.linalg.norm( exact['y'].values - tossi['y'].values ) )
    

def plotSpiralMethodsHeight():
    ## load data
    exact = pd.read_csv(f'.\\data\\spiralExact.csv', index_col=0)

    # plot
    plt.plot( exact.index, exact['y'], label="y", c='k', linestyle="--" ) # exact data
    plt.axvline( x=sp.tTossiEnd, c='grey', label='Training area', linewidth=.5 ) # show up to where was trained

    for method in sp.methodList:
        tossi = pd.read_csv( f'.\\data\\spiralTossi{method}.csv', index_col=0 )
        plt.plot( tossi.index, tossi['y'], label=method )
    plt.yscale('log')
    plt.legend()
    plt.show()
    





# die eigentliche plot-Funktion
def plotSpiralHeight( exactName, tossiName, outName ):
    imgId = getImgId( sp )

    ## load data
    exact = pd.read_csv(f'.\\data\\spiralExact{exactName}.csv', index_col=0)
    tossi = pd.read_csv(f'.\\data\\spiralTossi{tossiName}.csv', index_col=0)

    # ---------------------------------------------------------------
    # plot Spiral height
    # ---------------------------------------------------------------

    # plot
    plt.plot( exact.index, exact['y'], label="y", c='r' )
    plt.plot( tossi.index, tossi['y'], label="model", c='k', linestyle="--" )
    plt.axvline( x=sp.tTossiEnd, c='grey', label='Training area', linewidth=.5 ) # up to where was trained
    #plt.plot( exact.index, np.gradient(exact['y']), label="dy" )
    #plt.plot( tossi.index, np.gradient(tossi['y']), label="dyApprox" )
    plt.legend()
    plt.title( sp.method )
    
    plt.savefig( f'.\\plot\\spiralTossi2{outName}_i{imgId}.pgf' )
    plt.savefig( f'.\\plot\\spiralTossi2{outName}_i{imgId}.png' )
    plt.show()




    # ---------------------------------------------------------------
    # plot Spiral 3D
    # ---------------------------------------------------------------

    # plot
    fig = plt.figure()
    ax = fig.gca( projection='3d' )

    ax.plot( exact['x1'], exact['x2'], exact['y'], c='r', label='$x_k$')
    ax.plot( tossi['x1'], tossi['x2'], tossi['y'], c='k', linestyle='--', label='model' )

    #ax.set_xlim3d( -20, 20 )
    #ax.set_ylim3d( -50, 50 )
    #ax.set_zlim3d( 0, 50 )

    #ax.set_xticklabels([-20, 0, 20])
    #ax.set_yticklabels([-50, 0, 50])
    #ax.set_zticklabels([0, 25, 50])

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    #plt.title("Spiral")
    #ax.legend(loc='best')

    plt.savefig(  f'.\\plot\\spiralTossi3{outName}.pgf' )
    plt.savefig(  f'.\\plot\\spiralTossi3{outName}.png' )
    plt.show()
