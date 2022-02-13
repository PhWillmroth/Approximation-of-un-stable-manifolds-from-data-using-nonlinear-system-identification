# Extended Morrow-Patterson points
from numpy import cos, pi
from scipy.special import binom
from matplotlib import pyplot as plt

mu = 0

def x(m):
    return cos( (m*pi) / (mu+2) ) / cos( pi / (mu+2) )

def y(k, m):
    if m % 2 == 1:
        return cos(2*k*pi / (mu+3)) / cos(pi / (mu+3))
    else:
        return cos((2*k-1)*pi / (mu+3)) / cos(pi / (mu+3))

# main
# mu is the polyorder
def generateEMP(mymu):
    global mu
    mu = mymu
    assert mu % 2 == 0 # polyorder, must be even for our generation method

    emp = {(x(m), y(k, m)) for m in range(1, mu+2) for k in range(1, int(2+(mu/2)))}
    assert len(emp) == degOfFreedom(2, mu), "{} != {}".format(len(emp), degOfFreedom(2, mu))

    return emp

def degOfFreedom(n, mu):
    return binom(n+mu, mu)

def plotEMP(mu):
    emp = generateEMP(mu)
    for p in emp:
        plt.scatter(p[0], p[1], c="green")
    plt.show()