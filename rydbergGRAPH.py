#graph without the outliers
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.simplefilter(action='ignore')

hydrogen=(pd.read_csv('C:/Users/ztcpo/Desktop/LABORATORY/EXTENDED_PHYSICS_PROJECT/v2/hyd_outliers.csv'))
deuterium=(pd.read_csv('C:/Users/ztcpo/Desktop/LABORATORY/EXTENDED_PHYSICS_PROJECT/v2/deut_outliers.csv'))

y=hydrogen['rydberg']
x=deuterium['rydberg']
yerr=hydrogen['rerror']
xerr=deuterium['rerror']

fig = plt.figure(figsize=(13,12))                          
ax = fig.add_subplot(1,1,1)
ax.set_ylabel(r'$R_D\times 10^7 m^{-1}$',fontsize=30)
ax.set_xlabel(r'$R_H\times 10^7 m^{-1}$',fontsize=30)
ax.scatter(x,y,color='red',marker='x',linewidth=4,s=120)
ax.errorbar(x,y,yerr,xerr,color='k',linestyle='none',capsize=5)
ax.tick_params(axis="x", labelsize=25)
ax.tick_params(axis="y", labelsize=25)
a,b=np.polyfit(x,y,1)
#plt.axline(xy1=(10980000,11000000), slope=-0.9705662138394981)
plt.plot(x,(a*x)+b,color='red')
ax.plot()
print(a)

#mass of electron
"""def electron(alpha):
    #alpha=a*
    p=1.67262192369*10**-27
    n=1.67492749804*10**-27
    return (p*(1-alpha))/(((p*alpha)/(n+p))-1)"""

def electron(alpha):
    p=1.67262192369*10**-27
    n=1.67492749804*10**-27
    return (alpha-1)/((1/p)-(alpha/(n+p)))

print(electron(-a))
errorme=(0.159207148/0.9999)*electron(0.9999)
print("+-",errorme)
