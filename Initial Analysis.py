#code to run though our initial results, correct them from the
#central line and then find the incident angle (grating) and
#correct for this too.
#import packages
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.simplefilter(action='ignore')
#please edit this to access the file from wherever you have saved it
dataframe=(pd.read_csv('C:/Users/ztcpo/Desktop/LABORATORY/EXTENDED_PHYSICS_PROJECT/RAW_DEUTERIUM_1OD.csv'))

#turning this into a table with values all in degrees
dataframe['L(+)raw']=((dataframe['left_deg'])+(dataframe['left_min']/60))
dataframe['R(-)raw']=((dataframe['right_deg'])+(dataframe['right_min']/60))

#correcting this for the central line and modifying the RHS
central=359+(22/60)
dataframe['L(+)deg']=dataframe['L(+)raw']+(360-central)
dataframe['R(-)deg']=central-(dataframe['R(-)raw'])

#finding the value for theta_i(in degrees)
def theta_i(a_plus,a_min):
    return (np.sin(np.deg2rad(a_plus))+np.sin(np.deg2rad((a_min))))/(2-np.cos(np.deg2rad(a_plus))+np.cos(np.deg2rad(a_min)))

#theta_i values
av_theta_i=np.sum(theta_i(dataframe['L(+)deg'],dataframe['R(-)deg']))/3
print((theta_i(dataframe['L(+)deg'],dataframe['R(-)deg'])))
print('Grating Tilt:',av_theta_i)


dataframe['corrected_L(+)']=dataframe['L(+)deg']+av_theta_i
dataframe['corrected_R(-)']=dataframe['R(-)deg']-av_theta_i

results=dataframe[['colour','corrected_L(+)','corrected_R(-)','energy_lvl']]
results['difference']= results['corrected_L(+)']-results['corrected_R(-)']
print(results)

print("angles for each colour with associated error:")
results['average']=(results['corrected_L(+)']+results['corrected_R(-)'])/2
results['error']=results['difference']/2
final=results[['colour','energy_lvl','average','error']]

#Grating spacing Wavelengths and Rydberg constants
#we can use the original equation m*l=d*sin(theta)

#correcting the Na data
central=359+(22/60)
correction=360-central
n=1.00027 #refractive index for air
left=19+(41/60)+correction-av_theta_i
right=19/(40/60)+correction -av_theta_i

print(f"Na angle for D1:",{left},"and D2:",{right})

def grating(theta,m,l):
    return (m*l)/(n*np.sin(np.deg2rad(theta)))

print("Grating spacing D1:",grating(left,1,588.99*10**-9))
print("Grating spacing D2:",grating(right,1,589.59*10**-9))

grating_spacing = ((grating(left,1,588.99*10**-9)+grating(right,1,589.59*10**-9))/2)
unc_grating= (grating(left,1,588.99*10**-9)-grating(right,1,589.59*10**-9))/2
print("grating spacing=",grating_spacing,"+-",unc_grating)

#now we can find the wavelengths of the hydrogen and the deuterium emission lines
def wavelength(m,theta):
    return (n*1.69*10**-6*np.sin(np.deg2rad(theta)))/m

final['wavelength']=wavelength(1,final['average'])
final['wavelength_error']=(final['error']/final['average'])*final['wavelength']

def rydberg(lmbda, n_initial):
    return (1/lmbda) / ((1/(2**2)) - 1/(n_initial**2))

energy=((1/(2**2)) - (1/(final['energy_lvl'])**2))
inv_lamb=1/(final['wavelength']*10**6)
final['rydberg']=rydberg(final['wavelength'],final['energy_lvl'])
print(final)

#gfg_csv_data = final.to_csv('DEUTERIUM_ANALYSISV2.csv', index = False)
#print('\nCSV String:\n', gfg_csv_data)


y_error=((final['wavelength_error'])/(final['wavelength']))*inv_lamb

fig = plt.figure(figsize=(12,12))                          
ax = fig.add_subplot(1,1,1)
ax.set_ylabel(r'$\frac{1}{\lambda} \mu m^{-1}$',fontsize=55)
ax.set_xlabel(r'$\frac{1}{2^{2}}-\frac{1}{n^{2}}$',fontsize=55)
ax.xaxis.set_ticks(np.arange(0.13,0.24,0.02))
ax.plot(energy,inv_lamb,color='red',marker='o',markersize=15,linewidth=3)
ax.errorbar(energy,inv_lamb,yerr=y_error,color='k',linestyle='none',capsize=5)
ax.tick_params(axis='both', labelsize=35)
ax.set_xlim(left=0.13,right=0.22)
ax.set_ylim(top=2.4,bottom=1.45)
#ax.legend(fontsize=25)
ax.plot()


"""#finding the mass of the electron
h = 6.62607015 * 10**-34
c = 299792458
e0 = 8.8541878128 * 10**-12
e = 1.60217663 * 10**-19
mp = 1.67262192369 
mn=1.67492749804 * 10**-27
nuc=mn+mp
df=(pd.read_csv('C:/Users/ztcpo/Desktop/LABORATORY/EXTENDED_PHYSICS_PROJECT/v2/HYDROGEN_ANALYSISV2.csv'))
ryd=10997525.16
reduced_mass = 8 * df['rydberg'] * h**3 * c * e0**2 * e**-4
#reduced_mass_err = reduced_mass * rh_err / rh
electron_mass = (reduced_mass * nuc) / (nuc - reduced_mass)
#electron_mass_err = electron_mass * reduced_mass_err / reduced_mass
print(electron_mass)
"""