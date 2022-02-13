# Padua points
from numpy import cos, pi, array
from scipy.special import binom
from matplotlib import pyplot as plt

n = 2 # state dim (is always 2)
mu = 0 # polyorder (is updated below)

def x(m):
    return cos( pi * (m-1) / mu )

def y(k, m):
    if m % 2 == 1:
        return cos(2 * pi * (k-1) / (mu+1))
    else:
        return cos( (2*k-1) * pi / (mu+1) )

# main
# mymu is the polyorder
def generatePP(mymu):
    global mu
    mu = mymu
    assert mu % 2 == 0 # polyorder, must be even for our generation method

    pp = {(x(m), y(k, m)) for m in range(1, mu+2) for k in range(1, 2 + int(mu/2))}
    assert len(pp) == degOfFreedom(mu), "{} != {}".format(len(pp), degOfFreedom(mu))

    return [array(i) for i in pp]

def degOfFreedom(mu):
    return binom(n+mu, mu)

def plotPP(mu):
    pp = generatePP(mu)

    for p in pp:
        plt.scatter(p[0], p[1], c="green")
    plt.show()