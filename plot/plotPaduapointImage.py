import numpy as np
from matplotlib import pyplot as plt, rc_params

mu = 5
curvesamples = 5000

def lc(t):
    return np.cos(mu * t), np.cos((mu+1) * t)

# ----------------------------------------------
fig = plt.figure(figsize=(6,6))

# Plot Lisajous curve
t = np.linspace(0, np.pi, curvesamples)
a = np.array( lc(t) )
plt.plot(a[0], a[1], c='grey', alpha=0.75) 

# Plot Padua points
pp = { lc(np.pi * i / mu / (mu+1) ) for i in range(mu * (mu+1) + 1) }
b = np.array(list(pp)).T
plt.scatter(b[0], b[1], c='darkgreen')


plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])

#rc_params['figure.figsize'] = 1,1

plt.savefig('.\\plot\\padua.png')
plt.savefig('.\\plot\\padua.pgf')
plt.show()