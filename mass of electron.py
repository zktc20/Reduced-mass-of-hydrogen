import pandas as pd
import numpy as np
h = 6.62607015 * 10**-34
c = 299792458
e0 = 8.8541878128 * 10**-12
e = 1.60217663 * 10**-19
mp = 1.67262192369 
mn=1.67492749804 * 10**-27
nuc=mn+mp
df=(pd.read_csv('C:/Users/ztcpo/Desktop/LABORATORY/EXTENDED_PHYSICS_PROJECT/v2/rydd.csv'))
reduced_mass = 8 * df * h**3 * c * e0**2 * e**-4
#reduced_mass_err = reduced_mass * rh_err / rh
electron_mass = (reduced_mass * nuc) / (nuc - reduced_mass)
#electron_mass_err = electron_mass * reduced_mass_err / reduced_mass
print(electron_mass)
print(0.00465/np.sqrt(3))