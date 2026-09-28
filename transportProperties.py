# %%
from builtins import print
import os
from zipfile import Path
import collisionIntegrals as ci
import pandas as pd
import numpy as np

# Viscosity of component 1 (Kg/m*s) [use a factor of 0.1 to convert from g/cm*s to Kg/m*s]
# Constant factor for viscosity in g/cm*s: (5/16)*(1/sqrt(pi))*sqrt(k_boltzman/N_av)*sqrt(1000g)*(100cm)*(10**16Arm**2) = 0.00002669565
etha_c1 = 0.1 * (5/16) * (np.sqrt(1000)) * (10**16) * ((ci.c1_mw * ci.k_boltzman *
                                                        ci.T) / (np.pi * ci.N_Av))**(1/2) * (ci.f_h_c1) / ((ci.sig_c1**2)*ci.omega_22_c1)
# Viscosity of component 2 (g/cm*s)  [use a factor of 0.1 to convert from g/cm*s to Kg/m*s]
etha_c2 = 0.1 * (5/16) * (np.sqrt(1000)) * (10**16) * ((ci.c2_mw * ci.k_boltzman *
                                                        ci.T) / (np.pi * ci.N_Av))**(1/2) * (ci.f_h_c2) / ((ci.sig_c2**2)*ci.omega_22_c2)

# Thermal conductivity of component 1 [use a factor of 1000 to get W/m*K]
lambda_c1 = 1000 * (15/4) * (etha_c1 * ci.R_univ /
                             ci.c1_mw) * (ci.f_l_c1 / ci.f_h_c1)
# Thermal conductivity of component 2 [use a factor of 1000 to get W/m*K]
lambda_c2 = 1000 * (15/4) * (etha_c2 * ci.R_univ /
                             ci.c2_mw) * (ci.f_l_c2 / ci.f_h_c2)

# Self diffusion coefficient of component 1
p = 101325  # Pressure in Pa [Kg/m*s^2]
Dself_c1 = (3/8) * np.sqrt(1000) * (1e9)**2 * np.sqrt(((ci.k_boltzman * ci.T)**3) /
                                                      (np.pi * (ci.c1_mw/ci.N_Av))) * (ci.f_D_c1/(p * ci.omega_11_c1 * ci.sig_c1**2))
# Self diffusion coefficient of component 2
Dself_c2 = (3/8) * np.sqrt(1000) * (1e9)**2 * np.sqrt(((ci.k_boltzman * ci.T)**3) /
                                                      (np.pi * (ci.c2_mw/ci.N_Av))) * (ci.f_D_c2/(p * ci.omega_11_c2 * ci.sig_c2**2))

#################################################################################################################################################################################
# Viscosity of the mixture (Kg/m*s)
etha_12 = 0.1 * (5/16) * (np.sqrt(1000)) * (10**16) * np.sqrt(((2*ci.c1_mw*ci.c2_mw)/(ci.N_Av *
                                                                                      (ci.c1_mw+ci.c2_mw)))*(ci.k_boltzman*ci.T/np.pi)) * (1 / ((ci.sig_12_mix**2) * ci.omega_22))

X_etha = (ci.c1_mf**2)/etha_c1 + (2*ci.c1_mf*ci.c2_mf) / \
    etha_12 + (ci.c2_mf**2)/etha_c2

Y_etha = (3/5 * ci.A_r) * ((ci.c1_mf**2 / etha_c1) * (ci.c1_mw/ci.c2_mw) + (2 * ci.c1_mf * ci.c2_mf / etha_12) *
                           ((ci.c1_mw + ci.c2_mw)**2/(4*ci.c1_mw*ci.c2_mw))*(etha_12**2/(etha_c1*etha_c2)) + (ci.c2_mf**2/etha_c2)*(ci.c2_mw/ci.c1_mw))

Z_etha = (3/5 * ci.A_r) * (((ci.c1_mf**2)*(ci.c1_mw/ci.c2_mw)) + 2*ci.c1_mf*ci.c2_mf*(((ci.c1_mw+ci.c2_mw)
                                                                                       ** 2/(4*ci.c1_mw*ci.c2_mw))*(etha_12/etha_c1 + etha_12/etha_c2) - 1) + (ci.c2_mf**2)*(ci.c2_mw/ci.c1_mw))

etha_mix = (1 + Z_etha)/(X_etha + Y_etha)
#################################################################################################################################################################################
# Thermal conductivity of the mixture [use a factor of 1000 to get W/m*K]
C = ci.c2_mw/ci.c1_mw
A = np.sqrt(2)/(8*(1 + 1.8*C)**2) * (ci.omega_11/ci.omega_22_c2)
B = 10*A*(1 + 1.8*C + 3*C**2) - 1
DELTA = (1.3*((6*ci.C_r - 5)**2)*A*ci.c1_mf)/(1 + B*ci.c1_mf)
lambda_12 = 75/64 * np.sqrt(1000) * 1e18 * np.sqrt((((ci.k_boltzman)**3)*ci.T*ci.N_Av*(
    ci.c1_mw + ci.c2_mw)) / (2*np.pi*ci.c1_mw*ci.c2_mw)) * (1 + DELTA)/(ci.omega_22*ci.sig_12_mix**2)
#lambda_12 =  (15/4) * 1000 * ci.k_boltzman * ci.N_Av  * ((ci.c1_mw + ci.c2_mw)/(2*ci.c1_mw*ci.c2_mw)) * etha_12 

X_lambda = (ci.c1_mf**2)/lambda_c1 + (2*ci.c1_mf*ci.c2_mf) / \
    lambda_12 + (ci.c2_mf**2)/lambda_c2

U1 = (4/15*ci.A_r) - 1/12*(12/5*ci.B_r + 1)*(ci.c1_mw/ci.c2_mw) + \
    1/2*((ci.c1_mw - ci.c2_mw)**2 / (ci.c1_mw*ci.c2_mw))

U2 = (4/15*ci.A_r) - 1/12*(12/5*ci.B_r + 1)*(ci.c2_mw/ci.c1_mw) + \
    1/2*((ci.c2_mw - ci.c1_mw)**2 / (ci.c1_mw*ci.c2_mw))

UY = (4/15*ci.A_r)*((ci.c1_mw + ci.c2_mw)**2 / (4*ci.c1_mw*ci.c2_mw))*(lambda_12**2/(lambda_c1*lambda_c2)) - \
    1/12*(12/5*ci.B_r + 1) - (5/(32*ci.A_r))*(12/5*ci.B_r - 5) * \
    ((ci.c1_mw - ci.c2_mw)**2 / (ci.c1_mw*ci.c2_mw))

UZ = (4/15*ci.A_r)*((ci.c1_mw + ci.c2_mw)**2 / (4*ci.c1_mw*ci.c2_mw)) * \
    ((lambda_12/lambda_c1 + lambda_12/lambda_c2) - 1) - 1/12*(12/5*ci.B_r + 1)

Y_lambda = ((ci.c1_mf**2)/lambda_c1)*U1 + ((2*ci.c1_mf*ci.c2_mf) /
                                           lambda_12)*UY + ((ci.c2_mf**2)/lambda_c2)*U2

Z_lambda = (ci.c1_mf**2)*U1 + (2*ci.c1_mf*ci.c2_mf)*UZ + (ci.c2_mf**2)*U2

lambda_mix = (1 + Z_lambda)/(X_lambda + Y_lambda)
#################################################################################################################################################################################
# Mass diffusivity of the mixture [use a factor of 0.1 to get  m^2/s]
#D_mix = 0.1 * 0.002628 * np.sqrt(((ci.T**3)*(ci.c1_mw + ci.c2_mw))/(2*ci.c1_mw*ci.c2_mw)) * ci.f_D/(p*(ci.sig_12_mix**2)*ci.omega_11)
D_mix = 3/8 * np.sqrt(1000) * 1e18 * np.sqrt((((ci.k_boltzman*ci.T)**3)*ci.N_Av*(ci.c1_mw + ci.c2_mw)
                                              ) / (2*np.pi*ci.c1_mw*ci.c2_mw)) * (1 + DELTA)/(ci.omega_11*p*ci.sig_12_mix**2)
#################################################################################################################################################################################
# Thermal diffusion ratio of the mixture [-]
S_1 = ((ci.c1_mw + ci.c2_mw)/(2*ci.c2_mw))*(lambda_12/lambda_c1) - (15/(4*ci.A_r))*((ci.c2_mw - ci.c1_mw)/(2*ci.c1_mw)) - 1
S_2 = ((ci.c2_mw + ci.c1_mw)/(2*ci.c1_mw))*(lambda_12/lambda_c2) - (15/(4*ci.A_r))*((ci.c1_mw - ci.c2_mw)/(2*ci.c2_mw)) - 1

KT_mix= ((ci.c1_mf*ci.c2_mf)*(S_1*ci.c1_mf - S_2*ci.c2_mf)*(6*ci.C_r - 5))/((6*lambda_12)*(X_lambda + Y_lambda))
#################################################################################################################################################################################
# Thermal diffusion factor of the mixture [-]
Q_12 = 15*(((ci.c1_mw - ci.c2_mw)/(ci.c1_mw + ci.c2_mw))**2) * (5/2 - 6/5*ci.B_r) + ((4*ci.c1_mw*ci.c2_mw*ci.A_r)/(ci.c1_mw + ci.c2_mw)**2)*(11 - 12/5*ci.B_r) + \
    8/5*((ci.c1_mw + ci.c2_mw)/np.sqrt(ci.c1_mw*ci.c2_mw))*(((ci.sig_c1*ci.sig_c2) /
                                                             (ci.sig_12_mix**2))**2)*((ci.omega_22_c1*ci.omega_22_c2)/(ci.omega_11)**2)

S_1 = (ci.c1_mw/ci.c2_mw)*np.sqrt((2*ci.c2_mw)/(ci.c1_mw + ci.c2_mw))*((ci.omega_22_c1/ci.omega_11)*(ci.sig_c1/ci.sig_12_mix) **
                                                                       2) - (4*ci.c1_mw*ci.c2_mw*ci.A_r)/(ci.c1_mw + ci.c2_mw)**2 - 15/2*((ci.c2_mw*(ci.c2_mw - ci.c1_mw))/(ci.c1_mw + ci.c2_mw)**2)
S_2 = (ci.c2_mw/ci.c1_mw)*np.sqrt((2*ci.c1_mw)/(ci.c2_mw + ci.c1_mw))*((ci.omega_22_c2/ci.omega_11)*(ci.sig_c2/ci.sig_12_mix) **
                                                                       2) - (4*ci.c2_mw*ci.c1_mw*ci.A_r)/(ci.c2_mw + ci.c1_mw)**2 - 15/2*((ci.c1_mw*(ci.c1_mw - ci.c2_mw))/(ci.c2_mw + ci.c1_mw)**2)

Q_1 = (2/(ci.c2_mw*(ci.c1_mw + ci.c2_mw)))*np.sqrt(2*ci.c2_mw/(ci.c1_mw + ci.c2_mw)) * (ci.omega_22_c1/ci.omega_11) * \
    ((ci.sig_c1/ci.sig_12_mix)**2) * ((5/2 - 6/5*ci.B_r) *
                                      ci.c1_mw**2 + 3*ci.c2_mw**2 + 8/5*ci.c1_mw*ci.c2_mw*ci.A_r)
Q_2 = (2/(ci.c1_mw*(ci.c2_mw + ci.c1_mw)))*np.sqrt(2*ci.c1_mw/(ci.c2_mw + ci.c1_mw)) * (ci.omega_22_c2/ci.omega_11) * \
    ((ci.sig_c2/ci.sig_12_mix)**2) * ((5/2 - 6/5*ci.B_r) *
                                      ci.c2_mw**2 + 3*ci.c1_mw**2 + 8/5*ci.c2_mw*ci.c1_mw*ci.A_r)

kappa_2 = 0
# kappa_2  is a correction term of higher approximation but it is negliglibe compared with experimental uncertainty
alpha_T = (6*ci.C_r - 5)*((ci.c1_mf*S_1 - ci.c2_mf*S_2)/((ci.c1_mf**2)
                                                         * Q_1 + (ci.c2_mf**2)*Q_2 + ci.c1_mf*ci.c2_mf*Q_12))*(1 + kappa_2)

print('Temperature [K]')
print(ci.T)
print('Mole fraction of component 1: ' + str(ci.c1))
print(ci.c1_mf)
print('Reduced temperature of the ' +
      str(ci.c1) + '-' + str(ci.c2) + ' Mixture')
print(ci.T_r)
print('Viscosity of ' + str(ci.c1) + ' (Kg/m*s)')
print(etha_c1)
print('Viscosity of ' + str(ci.c2) + ' (Kg/m*s)')
print(etha_c2)
print('Thermal conductivity of ' + str(ci.c1) + ' (W/m*K)')
print(lambda_c1)
print('Thermal conductivity of ' + str(ci.c2) + ' (W/m*K)')
print(lambda_c2)
print('Self diffusivity of ' + str(ci.c1) + ' at P=101325Pa [Kg/m*s^2]')
print(Dself_c1)
print('Self diffusivity of ' + str(ci.c2) + ' at P=101325Pa [Kg/m*s^2]')
print(Dself_c2)
print('Viscosity of the mixture ' + str(ci.c1) + '-' + str(ci.c2) +
      ' at P=101325Pa [Kg/m*s^2] and T=' + str(ci.T) + 'K [Kg/ms]')
print(etha_mix)
print('Thermal conductivity of the mixture ' + str(ci.c1) + '-' +
      str(ci.c2) + ' at P=101325Pa [Kg/m*s^2] and T=' + str(ci.T) + 'K [W/Ks]')
print(lambda_mix)
print('Mass diffusivity of the mixture ' + str(ci.c1) + '-' +
      str(ci.c2) + ' at P=101325Pa [Kg/m*s^2] and T=' + str(ci.T) + 'K [m^2/s]')
print(D_mix)
print('Thermal diffusion factor of the mixture ' + str(ci.c1) + '-' +
      str(ci.c2) + ' at P=101325Pa [Kg/m*s^2] and T=' + str(ci.T) + 'K [-]')
print(alpha_T)


if not os.path.exists("./transportPropertiesCSV/"):
      os.mkdir("./transportPropertiesCSV/")
else:
      pass


viscosityMix = {'Mole fraction of component 1: ' + str(ci.c1):list(ci.c1_mf), str(ci.c1) + '-' + str(ci.c2) +
      ' mixture at T=' + str(ci.T) + 'K':list(etha_mix)}
df_viscosityMix=pd.DataFrame(viscosityMix, columns=['Mole fraction of component 1: ' + str(ci.c1), str(ci.c1) + '-' + str(ci.c2) +
      ' mixture at T=' + str(ci.T) + 'K'])
df_viscosityMix.to_csv('transportPropertiesCSV/' + str(ci.c1) + '-' + str(ci.c2) + '-T=' + str(ci.T) + 'mu-viscosityMix.csv', index=False)

conductivityMix = {'Mole fraction of component 1: ' + str(ci.c1):list(ci.c1_mf), str(ci.c1) + '-' + str(ci.c2) +
      ' mixture at T=' + str(ci.T) + 'K':list(lambda_mix)}
df_conductivityMix=pd.DataFrame(conductivityMix, columns=['Mole fraction of component 1: ' + str(ci.c1), str(ci.c1) + '-' + str(ci.c2) +
      ' mixture at T=' + str(ci.T) + 'K'])
df_conductivityMix.to_csv('transportPropertiesCSV/' + str(ci.c1) + '-' + str(ci.c2) + '-T=' + str(ci.T) + 'K-conductivityMix.csv', index=False)

diffusivityMix = {'Mole fraction of component 1: ' + str(ci.c1):list(ci.c1_mf), str(ci.c1) + '-' + str(ci.c2) +
      ' mixture at T=' + str(ci.T) + 'K':list(D_mix)}
df_diffusivityMix=pd.DataFrame(diffusivityMix, columns=['Mole fraction of component 1: ' + str(ci.c1), str(ci.c1) + '-' + str(ci.c2) +
      ' mixture at T=' + str(ci.T) + 'K'])
df_diffusivityMix.to_csv('transportPropertiesCSV/' + str(ci.c1) + '-' + str(ci.c2) + '-T=' + str(ci.T) + 'D12-diffusivityMix.csv', index=False)

thermaldiffMix = {'Mole fraction of component 1: ' + str(ci.c1):list(ci.c1_mf), str(ci.c1) + '-' + str(ci.c2) +
      ' mixture at T=' + str(ci.T) + 'K':list(alpha_T)}
df_thermaldiffMix=pd.DataFrame(thermaldiffMix, columns=['Mole fraction of component 1: ' + str(ci.c1), str(ci.c1) + '-' + str(ci.c2) +
      ' mixture at T=' + str(ci.T) + 'K'])
df_thermaldiffMix.to_csv('transportPropertiesCSV/' + str(ci.c1) + '-' + str(ci.c2) + '-T=' + str(ci.T) + 'KT-thermaldiffMix.csv', index=False)
# %%
