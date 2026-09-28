# %%
import pandas as pd
import numpy as np

#############################################################
# Initializing variables inside if statements
# For component 1
X_22_c1 = None
Y_22_c1 = None
d1X_22_c1 = None
d1Y_22_c1 = None
d2X_22_c1 = None
d2Y_22_c1 = None
d1_omega_22_c1 = None
d2_omega_22_c1 = None
omega_22_c1 = None
X_11_c1 = None
Y_11_c1 = None
d1X_11_c1 = None
d1Y_11_c1 = None
d2X_11_c1 = None
d2Y_11_c1 = None
d1_omega_11_c1 = None
d2_omega_11_c1 = None
omega_11_c1 = None
# For component 2
X_22_c2 = None
Y_22_c2 = None
d1X_22_c2 = None
d1Y_22_c2 = None
d2X_22_c2 = None
d2Y_22_c2 = None
d1_omega_22_c2 = None
d2_omega_22_c2 = None
omega_22_c2 = None
X_11_c2 = None
Y_11_c2 = None
d1X_11_c2 = None
d1Y_11_c2 = None
d1_omega_11_c2 = None
d2_omega_11_c2 = None
omega_11_c2 = None
# For Mixture
X_22 = None
Y_22 = None
d1X_22 = None
d1Y_22 = None
d1_omega_22 = None
d2_omega_22 = None
omega_22 = None
X_11 = None
Y_11 = None
d1X_11 = None
d1Y_11 = None
d1_omega_11 = None
d2_omega_11 = None
omega_11 = None
##############################################################################

# Reading csv file with thermophysical gas data.
nobleGasData = pd.read_csv("nobleGasProperties.csv", header=0)

# Universal constants
h = 6.626176e-34  # Planck constant [J*s]
k_boltzman = 1.380662e-23  # Boltzman constant [J/K]
N_Av = 6.022045e23  # Avogadro constant [1/mol]
R_univ = 8.31441  # Universal gas constant [J/(mol*K)]


# Binary mixture species input
print('Select the first component (Heavier) from the list: He, Ne, Ar, Kr, Xe')
c1 = input()
# print(c1)
print('Select the second component (Lighter) from the list: He, Ne, Ar, Kr, Xe')
c2 = input()
# print(c2)
print('Set the mole fraction of the mixture main component ' + c1 + ' (Value between 0 and 1)')
print('Note: Mole fraction of the mixture main component is set as an arange list: c1_mf = np.arange(0, 1.0, 0.1)')
#c1_mf = float(input())
c1_mf = np.arange(0, 1.1, 0.1)  # np.arange(0, 1.05, 0.05)
#c1_mf = np.arange(0.25,1.0,0.25)  # np.arange(0, 1.05, 0.05)
c2_mf = 1 - c1_mf

# Create the arrays with the selected noble gas species in the previous step
nobleGasData1 = nobleGasData[nobleGasData['Gas'].str.match(c1)]
nobleGasArray1 = nobleGasData1.to_numpy()

nobleGasData2 = nobleGasData[nobleGasData['Gas'].str.match(c2)]
nobleGasArray2 = nobleGasData2.to_numpy()

c1_mw = nobleGasArray1[0][1]
c2_mw = nobleGasArray2[0][1]
mix_mw = c1_mf*c1_mw + c2_mf*c2_mw
print('Set the mean temperature value [K]')
T = float(input())
#T = np.arange(373,593,50)
#T = np.array([273.15,293.15,313.15,333.15,353.15,373.15,423.15,473.15,523.15,573.15])
#T = np.array([50, 100, 150, 200, 250, 300])
#T = np.array([50, 273.15, 373.15, 623.15, 973.15, 2273.15])
#T = np.array([1773.15, 2273.5, 2773.15,3273.15])
#T = np.array([973.15, 1073.15, 1173.15, 1273.15])
#T=np.arange(253.15, 328.15, 5)
##################################################################################################################

# Calculate collision integrals for component 1
# Reduced temperature calculation for component 1
ek_c1 = nobleGasArray1[0][3]
T_r_c1 = T/ek_c1

# Colission integral coefficients
C6_r_c1 = nobleGasArray1[0][4]
rho_r_c1 = nobleGasArray1[0][5]
V_r_c1 = nobleGasArray1[0][6]
alpha_10_c1 = np.log(V_r_c1/10)
alpha_c1 = np.log(V_r_c1) - np.log(T_r_c1)
F_r_c1 = 0.9543 + 0.00124 * T_r_c1
sig_c1 = nobleGasArray1[0][2]

# colision integral omega(2,2)

if T_r_c1 <= 1.2:
    a1 = 0.18
    a2 = 0
    a3 = -1.20407 - 0.195866*(C6_r_c1**(-1/3))
    a4 = -9.86374 + 20.2221*(C6_r_c1**(-1/3))
    a5 = 16.6295 - 31.3613*(C6_r_c1**(-1/3))
    a6 = -6.73805 + 12.6611*(C6_r_c1**(-1/3))

    X_22_c1 = (1.1943 * ((C6_r_c1/T_r_c1)**(1/3)))
    Y_22_c1 = (1 + a1*(T_r_c1**(1/3)) + a2*(T_r_c1**(2/3)) + a3 *
               T_r_c1 + a4*(T_r_c1**(4/3)) + a5*(T_r_c1**(5/3)) + a6*(T_r_c1**2))
    # First derivative of X and Y
    d1X_22_c1 = - (1.1943 * ((C6_r_c1/T_r_c1)**(1/3)))/(3*T_r_c1)
    d1Y_22_c1 = (a1/3)*(T_r_c1)**(-2/3) + (5*a5/3)*(T_r_c1)**(2/3) + (2*a2/3) * \
        (T_r_c1)**(-1/3) + (4*a4/3)*(T_r_c1)**(1/3) + (2*a6)*T_r_c1 + a3
    # Second derivative of X/ and Y
    d2X_22_c1 = (4*1.1943 * (C6_r_c1/T_r_c1)**(1/3))/(9*(T_r_c1**2))
    d2Y_22_c1 = (-2*a1/9)*(T_r_c1)**(-5/3) - (2*a2/9)*(T_r_c1)**(-4/3) + \
        (4*a4/9)*(T_r_c1)**(-2/3) + (10*a5/9)*(T_r_c1)**(-1/3) + 2*a6

    omega_22_c1 = X_22_c1*Y_22_c1
    # First derivative of ln(omega_22)
    d1_omega_22_c1 = d1X_22_c1/X_22_c1 + d1Y_22_c1/Y_22_c1
    # second derivative of ln(omega_22)
    d2_omega_22_c1 = (X_22_c1*d2X_22_c1 - d1X_22_c1**2) / \
        (X_22_c1**2) + (Y_22_c1*d2Y_22_c1 - d1Y_22_c1**2)/(Y_22_c1**2)

elif T_r_c1 > 1.2 and T_r_c1 < 10:
    omega_22_c1 = np.exp(0.46641 - 0.56991 * np.log(T_r_c1) + 0.19591*(np.log(T_r_c1))
                         ** 2 - 0.03879*(np.log(T_r_c1))**3 + 0.00259*(np.log(T_r_c1))**4)
    # First derivative of ln(omega_22)
    d1_omega_22_c1 = (0.01036*(np.log(T_r_c1)**3) - 0.11637 *
                      (np.log(T_r_c1)**2) + 0.39182*(np.log(T_r_c1)) - 0.56991)/T_r_c1
    # second derivative of ln(omega_22)
    d2_omega_22_c1 = (-0.01036*(np.log(T_r_c1)**3) + 0.14745 *
                      (np.log(T_r_c1)**2) - 0.62456*(np.log(T_r_c1)) + 0.96173)/(T_r_c1**2)

elif T_r_c1 >= 10:
    b1 = 0
    b2 = -33.0838 + ((alpha_10_c1 * rho_r_c1)**(-2)) * \
        (20.0862 + 72.1059/alpha_10_c1 + (8.27648/alpha_10_c1)**2)
    b3 = 101.571 - ((alpha_10_c1 * rho_r_c1)**(-2)) * \
        (56.4472 + 286.393/alpha_10_c1 + (17.7610/alpha_10_c1)**2)
    b4 = -87.7036 + ((alpha_10_c1 * rho_r_c1)**(-2)) * \
        (46.3130 + 277.146/alpha_10_c1 + (19.0573/alpha_10_c1)**2)

    X_22_c1 = (rho_r_c1*alpha_c1)**2

    Y_22_c1 = (1.04 + b1*(np.log(T_r_c1)**(-1)) + b2*(np.log(T_r_c1)
               ** (-2)) + b3*(np.log(T_r_c1)**(-3)) + b4*(np.log(T_r_c1)**(-4)))
    # First derivative of X and Y
    d1X_22_c1 = (-2*((rho_r_c1)**2) *
                 (np.log(V_r_c1) - np.log(T_r_c1))) / T_r_c1
    d1Y_22_c1 = (-(b1*(np.log(T_r_c1)**3) + 2*b2*(np.log(T_r_c1)**2) +
                 3*b3*(np.log(T_r_c1)) + 4*b4))/(T_r_c1 * (np.log(T_r_c1)**5))
    # Second derivative of X and Y
    d2X_22_c1 = (-2*rho_r_c1**2) * (np.log(T_r_c1) -
                                    np.log(V_r_c1) - 1) / (T_r_c1**2)
    d2Y_22_c1 = (b1*(np.log(T_r_c1)+2)*np.log(T_r_c1)**3 + 2*b2*(np.log(T_r_c1) + 3)*np.log(T_r_c1)**2 + 3*b3 *
                 np.log(T_r_c1)**2 + 12*b3*np.log(T_r_c1) + 4*b4*np.log(T_r_c1) + 20*b4)/((T_r_c1**2) * (np.log(T_r_c1)**6))

    omega_22_c1 = X_22_c1*Y_22_c1
    # First derivative of ln(omega_22)
    d1_omega_22_c1 = d1X_22_c1/X_22_c1 + d1Y_22_c1/Y_22_c1
    # second derivative of ln(omega_22)
    d2_omega_22_c1 = (X_22_c1*d2X_22_c1 - d1X_22_c1**2) / \
        (X_22_c1**2) + (Y_22_c1*d2Y_22_c1 - d1Y_22_c1**2)/(Y_22_c1**2)
else:
    print()


# colision integral omega(1,1)
if T_r_c1 <= 1.2:
    c1 = 0
    c2 = 0
    c3 = 10.0161 - 10.5395 * C6_r_c1**(-1/3)
    c4 = -40.0394 + 46.0048 * C6_r_c1**(-1/3)
    c5 = 44.3202 - 53.0827 * C6_r_c1**(-1/3)
    c6 = -15.2912 + 18.8125 * C6_r_c1**(-1/3)

    X_11_c1 = 1.1874 * (C6_r_c1/T_r_c1)**(1/3)
    Y_11_c1 = (1 + c1 * (T_r_c1)**(1/3) + c2 * (T_r_c1)**(2/3) + c3 *
               T_r_c1 + c4 * (T_r_c1)**(4/3) + c5 * (T_r_c1)**(5/3) + c6 * (T_r_c1)**2)
    # First derivative of X and Y
    d1X_11_c1 = - (1.1874 * (C6_r_c1/T_r_c1)**(1/3))/(3*T_r_c1)
    d1Y_11_c1 = (c1/3)*(T_r_c1)**(-2/3) + (5*c5/3)*(T_r_c1)**(2/3) + (2*c2/3) * \
        (T_r_c1)**(-1/3) + (4*c4/3)*(T_r_c1)**(1/3) + (2*c6)*T_r_c1 + c3
    # Second derivative of X/ and Y
    d2X_11_c1 = (4*1.1874 * (C6_r_c1/T_r_c1)**(1/3))/(9*T_r_c1**2)
    d2Y_11_c1 = (-2*c1/9)*(T_r_c1)**(-5/3) - (2*c2/9)*(T_r_c1)**(-4/3) + \
        (4*c4/9)*(T_r_c1)**(-2/3) + (10*c5/9)*(T_r_c1)**(-1/3) + 2*c6

    omega_11_c1 = X_11_c1*Y_11_c1
    # First derivative of ln(omega_22)
    d1_omega_11_c1 = d1X_11_c1/X_11_c1 + d1Y_11_c1/Y_11_c1
    # second derivative of ln(omega_22)
    d2_omega_11_c1 = (X_11_c1*d2X_11_c1 - d1X_11_c1**2) / \
        (X_11_c1**2) + (Y_11_c1*d2Y_11_c1 - d1Y_11_c1**2)/(Y_11_c1**2)


elif T_r_c1 > 1.2 and T_r_c1 < 10:
    omega_11_c1 = np.exp(0.357588 - 0.472513 * np.log(T_r_c1) + 0.0700902 * (np.log(
        T_r_c1))**2 + 0.0165741 * (np.log(T_r_c1))**3 - 0.00592022 * (np.log(T_r_c1))**4)
    d1_omega_11_c1 = (-0.0236809*np.log(T_r_c1)**3 + 0.0049722 *
                      np.log(T_r_c1)**2 + 0.14018*np.log(T_r_c1) - 0.472513)/T_r_c1
    d2_omega_11_c1 = (0.236809*np.log(T_r_c1)**3 - 0.120765*np.log(T_r_c1)
                      ** 2 - 0.0407364*np.log(T_r_c1) + 0.612693)/(T_r_c1**2)
elif T_r_c1 >= 10:
    d2 = -267.00 + ((alpha_10_c1*rho_r_c1)**(-2)) * (201.57 +
                                                     174.672/alpha_10_c1 + (7.36916/alpha_10_c1)**2)
    d4 = (26.7 - (alpha_10_c1*rho_r_c1)**(-2) * (19.2265 +
          27.6938/alpha_10_c1 + (3.29559/alpha_10_c1)**2)) * 10**3
    d6 = (-8.9 + (alpha_10_c1*rho_r_c1)**(-2) * (6.31013 +
          10.2266/alpha_10_c1 + (2.33033/alpha_10_c1)**2)) * 10**5

    X_11_c1 = (rho_r_c1*alpha_c1)**2
    Y_11_c1 = 0.89 + d2/(T_r_c1**2) + d4/(T_r_c1**4) + d6/(T_r_c1**6)
    # First derivative of X and Y
    d1X_11_c1 = (-2*rho_r_c1**2) * ((np.log(V_r_c1) - np.log(T_r_c1)) / T_r_c1)
    d1Y_11_c1 = (-2) * (d2*T_r_c1**4 + 2*d4 * T_r_c1**2 + 3*d6) / (T_r_c1**7)
    # Second derivative of X and Y
    d2X_11_c1 = (-2*rho_r_c1**2) * (np.log(T_r_c1) -
                                    (np.log(V_r_c1) - 1) / (T_r_c1**2))
    d2Y_11_c1 = (6*d2*T_r_c1**4 + 20*d4*T_r_c1**2 + 42*d6)/(T_r_c1**8)

    omega_11_c1 = X_11_c1*Y_11_c1
    # First derivative of ln(omega_22)
    d1_omega_11_c1 = d1X_11_c1/X_11_c1 + d1Y_11_c1/Y_11_c1
    # second derivative of ln(omega_22)
    d2_omega_11_c1 = ((d2X_11_c1 - (d1X_11_c1)**2)/X_11_c1 + (d2Y_11_c1 -
                      (d1Y_11_c1)**2)/Y_11_c1) / (d1X_11_c1/X_11_c1 + d1Y_11_c1/Y_11_c1)
else:
    print()

# Recursion relations for component 1
A_r_c1 = omega_22_c1 / omega_11_c1
E_r_c1 = 1 + (T_r_c1/4) * d1_omega_22_c1
C_r_c1 = 1 + (T_r_c1/3) * d1_omega_11_c1
# 1 + 3*(C_r_c1 - C_r_c1**2) - (T_r_c1**2)*d2_omega_11_c1/3
B_r_c1 = 4*C_r_c1 - 3*C_r_c1**2 - 1/3*d2_omega_11_c1
H_r_c1 = (3*B_r_c1 + 6*C_r_c1 - 35/4) / (6*C_r_c1 - 5)
#F_r = omega_33/omega_11
f_h_c1 = 1 + (3/196) * ((8*E_r_c1 - 7)**2)
f_D_c1 = 1 + (1/8) * ((6*C_r_c1 - 5)**2) / (2*A_r_c1 + 5)
f_l_c1 = 1 + (1/42) * ((8*E_r_c1 - 7)**2)
########################################################################################################
# Calculate collision integrals for component 2
# Reduced temperature calculation for component 2
ek_c2 = nobleGasArray2[0][3]
T_r_c2 = T/ek_c2

# Colission integral coefficients
C6_r_c2 = nobleGasArray2[0][4]
rho_r_c2 = nobleGasArray2[0][5]
V_r_c2 = nobleGasArray2[0][6]
alpha_10_c2 = np.log(V_r_c2/10)
alpha_c2 = np.log(V_r_c2) - np.log(T_r_c2)
F_r_c2 = 0.9543 + 0.00124 * T_r_c2
sig_c2 = nobleGasArray2[0][2]

# colision integral omega(2,2)
if T_r_c2 <= 1.2:
    a1 = 0.18
    a2 = 0
    a3 = -1.20407 - 0.195866*(C6_r_c2**(-1/3))
    a4 = -9.86374 + 20.2221*(C6_r_c2**(-1/3))
    a5 = 16.6295 - 31.3613*(C6_r_c2**(-1/3))
    a6 = -6.73805 + 12.6611*(C6_r_c2**(-1/3))

    X_22_c2 = (1.1943 * ((C6_r_c2/T_r_c2)**(1/3)))
    Y_22_c2 = (1 + a1*(T_r_c2**(1/3)) + a2*(T_r_c2**(2/3)) + a3 *
               T_r_c2 + a4*(T_r_c2**(4/3)) + a5*(T_r_c2**(5/3)) + a6*(T_r_c2**2))
    # First derivative of X and Y
    d1X_22_c2 = - (1.1943 * ((C6_r_c2/T_r_c2)**(1/3)))/(3*T_r_c2)
    d1Y_22_c2 = (a1/3)*(T_r_c2)**(-2/3) + (5*a5/3)*(T_r_c2)**(2/3) + (2*a2/3) * \
        (T_r_c2)**(-1/3) + (4*a4/3)*(T_r_c2)**(1/3) + (2*a6)*T_r_c2 + a3
    # Second derivative of X/ and Y
    d2X_22_c2 = (4*1.1943 * (C6_r_c2/T_r_c2)**(1/3))/(9*(T_r_c2**2))
    d2Y_22_c2 = (-2*a1/9)*(T_r_c2)**(-5/3) - (2*a2/9)*(T_r_c2)**(-4/3) + \
        (4*a4/9)*(T_r_c2)**(-2/3) + (10*a5/9)*(T_r_c2)**(-1/3) + 2*a6

    omega_22_c2 = X_22_c2*Y_22_c2
    # First derivative of ln(omega_22)
    d1_omega_22_c2 = d1X_22_c2/X_22_c2 + d1Y_22_c2/Y_22_c2
    # second derivative of ln(omega_22)
    d2_omega_22_c2 = (X_22_c2*d2X_22_c2 - d1X_22_c2**2) / \
        (X_22_c2**2) + (Y_22_c2*d2Y_22_c2 - d1Y_22_c2**2)/(Y_22_c2**2)

elif T_r_c2 > 1.2 and T_r_c2 < 10:
    omega_22_c2 = np.exp(0.46641 - 0.56991 * np.log(T_r_c2) + 0.19591*(np.log(T_r_c2))
                         ** 2 - 0.03879*(np.log(T_r_c2))**3 + 0.00259*(np.log(T_r_c2))**4)
    # First derivative of ln(omega_22)
    d1_omega_22_c2 = (0.01036*(np.log(T_r_c2)**3) - 0.11637 *
                      (np.log(T_r_c2)**2) + 0.39182*(np.log(T_r_c2)) - 0.56991)/T_r_c2
    # second derivative of ln(omega_22)
    d2_omega_22_c2 = (-0.01036*(np.log(T_r_c2)**3) + 0.14745 *
                      (np.log(T_r_c2)**2) - 0.62456*(np.log(T_r_c2)) + 0.96173)/(T_r_c2**2)

else:
    b1 = 0
    b2 = -33.0838 + ((alpha_10_c2 * rho_r_c2)**(-2)) * \
        (20.0862 + 72.1059/alpha_10_c2 + (8.27648/alpha_10_c2)**2)
    b3 = 101.571 - ((alpha_10_c2 * rho_r_c2)**(-2)) * \
        (56.4472 + 286.393/alpha_10_c2 + (17.7610/alpha_10_c2)**2)
    b4 = -87.7036 + ((alpha_10_c2 * rho_r_c2)**(-2)) * \
        (46.3130 + 277.146/alpha_10_c2 + (19.0573/alpha_10_c2)**2)

    X_22_c2 = (rho_r_c2*alpha_c2)**2

    Y_22_c2 = (1.04 + b1*(np.log(T_r_c2)**(-1)) + b2*(np.log(T_r_c2)
               ** (-2)) + b3*(np.log(T_r_c2)**(-3)) + b4*(np.log(T_r_c2)**(-4)))
    # First derivative of X and Y
    d1X_22_c2 = (-2*((rho_r_c2)**2) *
                 (np.log(V_r_c2) - np.log(T_r_c2))) / T_r_c2
    d1Y_22_c2 = (-(b1*(np.log(T_r_c2)**3) + 2*b2*(np.log(T_r_c2)**2) +
                 3*b3*(np.log(T_r_c2)) + 4*b4))/(T_r_c2 * (np.log(T_r_c2)**5))
    # Second derivative of X and Y
    d2X_22_c2 = (-2*rho_r_c2**2) * (np.log(T_r_c2) -
                                    np.log(V_r_c2) - 1) / (T_r_c2**2)
    d2Y_22_c2 = (b1*(np.log(T_r_c2)+2)*np.log(T_r_c2)**3 + 2*b2*(np.log(T_r_c2) + 3)*np.log(T_r_c2)**2 + 3*b3 *
                 np.log(T_r_c2)**2 + 12*b3*np.log(T_r_c2) + 4*b4*np.log(T_r_c2) + 20*b4)/((T_r_c2**2) * (np.log(T_r_c2)**6))

    omega_22_c2 = X_22_c2*Y_22_c2
    # First derivative of ln(omega_22)
    d1_omega_22_c2 = d1X_22_c2/X_22_c2 + d1Y_22_c2/Y_22_c2
    # second derivative of ln(omega_22)
    d2_omega_22_c2 = (X_22_c2*d2X_22_c2 - d1X_22_c2**2) / \
        (X_22_c2**2) + (Y_22_c2*d2Y_22_c2 - d1Y_22_c2**2)/(Y_22_c2**2)


# colision integral omega(1,1)

if T_r_c2 <= 1.2:
    c2 = 0
    c2 = 0
    c3 = 10.0161 - 10.5395 * C6_r_c2**(-1/3)
    c4 = -40.0394 + 46.0048 * C6_r_c2**(-1/3)
    c5 = 44.3202 - 53.0827 * C6_r_c2**(-1/3)
    c6 = -15.2912 + 18.8125 * C6_r_c2**(-1/3)

    X_11_c2 = 1.1874 * (C6_r_c2/T_r_c2)**(1/3)
    Y_11_c2 = (1 + c2 * (T_r_c2)**(1/3) + c2 * (T_r_c2)**(2/3) + c3 *
               T_r_c2 + c4 * (T_r_c2)**(4/3) + c5 * (T_r_c2)**(5/3) + c6 * (T_r_c2)**2)
    # First derivative of X and Y
    d1X_11_c2 = - (1.1874 * (C6_r_c2/T_r_c2)**(1/3))/(3*T_r_c2)
    d1Y_11_c2 = (c2/3)*(T_r_c2)**(-2/3) + (5*c5/3)*(T_r_c2)**(2/3) + (2*c2/3) * \
        (T_r_c2)**(-1/3) + (4*c4/3)*(T_r_c2)**(1/3) + (2*c6)*T_r_c2 + c3
    # Second derivative of X/ and Y
    d2X_11_c2 = (4*1.1874 * (C6_r_c2/T_r_c2)**(1/3))/(9*T_r_c2**2)
    d2Y_11_c2 = (-2*c2/9)*(T_r_c2)**(-5/3) - (2*c2/9)*(T_r_c2)**(-4/3) + \
        (4*c4/9)*(T_r_c2)**(-2/3) + (10*c5/9)*(T_r_c2)**(-1/3) + 2*c6

    omega_11_c2 = X_11_c2*Y_11_c2
    # First derivative of ln(omega_22)
    d1_omega_11_c2 = d1X_11_c2/X_11_c2 + d1Y_11_c2/Y_11_c2
    # second derivative of ln(omega_22)
    d2_omega_11_c2 = (X_11_c2*d2X_11_c2 - d1X_11_c2**2) / \
        (X_11_c2**2) + (Y_11_c2*d2Y_11_c2 - d1Y_11_c2**2)/(Y_11_c2**2)


elif T_r_c2 > 1.2 and T_r_c2 < 10:
    omega_11_c2 = np.exp(0.357588 - 0.472513 * np.log(T_r_c2) + 0.0700902 * (np.log(
        T_r_c2))**2 + 0.0165741 * (np.log(T_r_c2))**3 - 0.00592022 * (np.log(T_r_c2))**4)
    d1_omega_11_c2 = (-0.0236809*np.log(T_r_c2)**3 + 0.0049722 *
                      np.log(T_r_c2)**2 + 0.14018*np.log(T_r_c2) - 0.472513)/T_r_c2
    d2_omega_11_c2 = (0.236809*np.log(T_r_c2)**3 - 0.120765*np.log(T_r_c2)
                      ** 2 - 0.0407364*np.log(T_r_c2) + 0.612693)/(T_r_c2**2)
else:
    d2 = -267.00 + (alpha_10_c2*rho_r_c2)**(-2) * (201.57 +
                                                   174.672/alpha_10_c2 + (7.3691/alpha_10_c2)**2)
    d4 = 26.7e3 - (alpha_10_c2*rho_r_c2)**(-2) * (19.2265 +
                                                  27.6938/alpha_10_c2 + (3.29559/alpha_10_c2)**2) * 10**3
    d6 = -8.9e5 + (alpha_10_c2*rho_r_c2)**(-2) * (6.31013 +
                                                  10.2266/alpha_10_c2 + (2.33033/alpha_10_c2)**2) * 10**5

    X_11_c2 = (rho_r_c2*alpha_c2)**2
    Y_11_c2 = 0.89 + d2/(T_r_c2**2) + d4/(T_r_c2**4) + d6/(T_r_c2**6)
    # First derivative of X and Y
    d1X_11_c2 = (-2*rho_r_c2**2) * ((np.log(V_r_c2) - np.log(T_r_c2)) / T_r_c2)
    d1Y_11_c2 = (-2) * (d2*T_r_c2**4 + 2*d4 * T_r_c2**2 + 3*d6) / (T_r_c2**7)
    # Second derivative of X and Y
    d2X_11_c2 = (-2*rho_r_c2**2) * (np.log(T_r_c2) -
                                    (np.log(V_r_c2) - 1) / (T_r_c2**2))
    d2Y_11_c2 = (6*d2*T_r_c2**4 + 20*d4*T_r_c2**2 + 42*d6)/(T_r_c2**8)

    omega_11_c2 = X_11_c2*Y_11_c2
    # First derivative of ln(omega_22)
    d1_omega_11_c2 = d1X_11_c2/X_11_c2 + d1Y_11_c2/Y_11_c2
    # second derivative of ln(omega_22)
    d2_omega_11_c2 = ((d2X_11_c2 - (d1X_11_c2)**2)/X_11_c2 + (d2Y_11_c2 -
                      (d1Y_11_c2)**2)/Y_11_c2) / (d1X_11_c2/X_11_c2 + d1Y_11_c2/Y_11_c2)


# Recursion relations for component 2
A_r_c2 = omega_22_c2 / omega_11_c2
E_r_c2 = 1 + (T_r_c2/4) * d1_omega_22_c2
C_r_c2 = 1 + (T_r_c2/3) * d1_omega_11_c2
# 1 + 3*(C_r_c2 - C_r_c2**2) - (T_r_c2**2)*d2_omega_11_c2/3
B_r_c2 = 4*C_r_c2 - 3*C_r_c2**2 - 1/3*d2_omega_11_c2
H_r_c2 = (3*B_r_c2 + 6*C_r_c2 - 35/4) / (6*C_r_c2 - 5)
#F_r = omega_33/omega_11
f_h_c2 = 1 + (3/196) * ((8*E_r_c2 - 7)**2)
f_D_c2 = 1 + (1/8) * (((6*C_r_c2 - 5)**2) * (2*A_r_c2 + 5)**(-1))
f_l_c2 = 1 + (1/42) * ((8*E_r_c2 - 7)**2)

###############################################################################################################
# Calculate collision integrals for mixture
# Reduced temperature calculation
ek_mix = np.sqrt(nobleGasArray1[0][3]*nobleGasArray2[0][3])
T_r = T/ek_mix

# Colission integral coefficients
C6_r = np.sqrt(nobleGasArray1[0][4]*nobleGasArray2[0][4])
rho_r = np.sqrt(nobleGasArray1[0][5]*nobleGasArray2[0][5])
V_r = np.sqrt(nobleGasArray1[0][6]*nobleGasArray2[0][6])
alpha_10 = np.log(V_r/10)
alpha = np.log(V_r) - np.log(T_r)
F_r = 0.9543 + 0.00124 * T_r
sig_12_mix = np.sqrt(nobleGasArray1[0][2]*nobleGasArray2[0][2])

# colision integral omega(2,2)
if T_r <= 1.2:
    a1 = 0.18
    a2 = 0
    a3 = -1.20407 - 0.195866*(C6_r**(-1/3))
    a4 = -9.86374 + 20.2221*(C6_r**(-1/3))
    a5 = 16.6295 - 31.3613*(C6_r**(-1/3))
    a6 = -6.73805 + 12.6611*(C6_r**(-1/3))

    X_22 = (1.1943 * ((C6_r/T_r)**(1/3)))
    Y_22 = (1 + a1*(T_r**(1/3)) + a2*(T_r**(2/3)) + a3*T_r +
            a4*(T_r**(4/3)) + a5*(T_r**(5/3)) + a6*(T_r**2))
    # First derivative of X and Y
    d1X_22 = - (1.1943 * ((C6_r/T_r)**(1/3)))/(3*T_r)
    d1Y_22 = (a1/3)*(T_r)**(-2/3) + (5*a5/3)*(T_r)**(2/3) + (2*a2/3) * \
        (T_r)**(-1/3) + (4*a4/3)*(T_r)**(1/3) + (2*a6)*T_r + a3
    # Second derivative of X/ and Y
    d2X_22 = (4*1.1943 * (C6_r/T_r)**(1/3))/(9*(T_r**2))
    d2Y_22 = (-2*a1/9)*(T_r)**(-5/3) - (2*a2/9)*(T_r)**(-4/3) + \
        (4*a4/9)*(T_r)**(-2/3) + (10*a5/9)*(T_r)**(-1/3) + 2*a6

    omega_22 = X_22*Y_22
    # First derivative of ln(omega_22)
    d1_omega_22 = d1X_22/X_22 + d1Y_22/Y_22
    # second derivative of ln(omega_22)
    d2_omega_22 = (X_22*d2X_22 - d1X_22**2)/(X_22**2) + \
        (Y_22*d2Y_22 - d1Y_22**2)/(Y_22**2)

elif T_r > 1.2 and T_r < 10:
    omega_22 = np.exp(0.46641 - 0.56991 * np.log(T_r) + 0.19591*(np.log(T_r))
                      ** 2 - 0.03879*(np.log(T_r))**3 + 0.00259*(np.log(T_r))**4)
    # First derivative of ln(omega_22)
    d1_omega_22 = (0.01036*(np.log(T_r)**3) - 0.11637 *
                   (np.log(T_r)**2) + 0.39182*(np.log(T_r)) - 0.56991)/T_r
    # second derivative of ln(omega_22)
    d2_omega_22 = (-0.01036*(np.log(T_r)**3) + 0.14745 *
                   (np.log(T_r)**2) - 0.62456*(np.log(T_r)) + 0.96173)/(T_r**2)

else:
    b1 = 0
    b2 = -33.0838 + ((alpha_10 * rho_r)**(-2))*(20.0862 +
                                                72.1059/alpha_10 + (8.27648/alpha_10)**2)
    b3 = 101.571 - ((alpha_10 * rho_r)**(-2))*(56.4472 +
                                               286.393/alpha_10 + (17.7610/alpha_10)**2)
    b4 = -87.7036 + ((alpha_10 * rho_r)**(-2))*(46.3130 +
                                                277.146/alpha_10 + (19.0573/alpha_10)**2)

    X_22 = (rho_r*alpha)**2

    Y_22 = (1.04 + b1*(np.log(T_r)**(-1)) + b2*(np.log(T_r)**(-2)) +
            b3*(np.log(T_r)**(-3)) + b4*(np.log(T_r)**(-4)))
    # First derivative of X and Y
    d1X_22 = (-2*((rho_r)**2) * (np.log(V_r) - np.log(T_r))) / T_r
    d1Y_22 = (-(b1*(np.log(T_r)**3) + 2*b2*(np.log(T_r)**2) + 3 *
              b3*(np.log(T_r)) + 4*b4))/(T_r * (np.log(T_r)**5))
    # Second derivative of X and Y
    d2X_22 = (-2*rho_r**2) * (np.log(T_r) - np.log(V_r) - 1) / (T_r**2)
    d2Y_22 = (b1*(np.log(T_r)+2)*np.log(T_r)**3 + 2*b2*(np.log(T_r) + 3)*np.log(T_r)**2 + 3*b3 *
              np.log(T_r)**2 + 12*b3*np.log(T_r) + 4*b4*np.log(T_r) + 20*b4)/((T_r**2) * (np.log(T_r)**6))

    omega_22 = X_22*Y_22
    # First derivative of ln(omega_22)
    d1_omega_22 = d1X_22/X_22 + d1Y_22/Y_22
    # second derivative of ln(omega_22)
    d2_omega_22 = (X_22*d2X_22 - d1X_22**2)/(X_22**2) + \
        (Y_22*d2Y_22 - d1Y_22**2)/(Y_22**2)


# colision integral omega(1,1)

if T_r <= 1.2:
    c2 = 0
    c2 = 0
    c3 = 10.0161 - 10.5395 * C6_r**(-1/3)
    c4 = -40.0394 + 46.0048 * C6_r**(-1/3)
    c5 = 44.3202 - 53.0827 * C6_r**(-1/3)
    c6 = -15.2912 + 18.8125 * C6_r**(-1/3)

    X_11 = 1.1874 * (C6_r/T_r)**(1/3)
    Y_11 = (1 + c2 * (T_r)**(1/3) + c2 * (T_r)**(2/3) + c3 * T_r +
            c4 * (T_r)**(4/3) + c5 * (T_r)**(5/3) + c6 * (T_r)**2)
    # First derivative of X and Y
    d1X_11 = - (1.1874 * (C6_r/T_r)**(1/3))/(3*T_r)
    d1Y_11 = (c2/3)*(T_r)**(-2/3) + (5*c5/3)*(T_r)**(2/3) + (2*c2/3) * \
        (T_r)**(-1/3) + (4*c4/3)*(T_r)**(1/3) + (2*c6)*T_r + c3
    # Second derivative of X/ and Y
    d2X_11 = (4*1.1874 * (C6_r/T_r)**(1/3))/(9*T_r**2)
    d2Y_11 = (-2*c2/9)*(T_r)**(-5/3) - (2*c2/9)*(T_r)**(-4/3) + \
        (4*c4/9)*(T_r)**(-2/3) + (10*c5/9)*(T_r)**(-1/3) + 2*c6

    omega_11 = X_11*Y_11
    # First derivative of ln(omega_22)
    d1_omega_11 = d1X_11/X_11 + d1Y_11/Y_11
    # second derivative of ln(omega_22)
    d2_omega_11 = (X_11*d2X_11 - d1X_11**2)/(X_11**2) + \
        (Y_11*d2Y_11 - d1Y_11**2)/(Y_11**2)


elif T_r > 1.2 and T_r < 10:
    omega_11 = np.exp(0.357588 - 0.472513 * np.log(T_r) + 0.0700902 * (np.log(T_r))
                      ** 2 + 0.0165741 * (np.log(T_r))**3 - 0.00592022 * (np.log(T_r))**4)
    d1_omega_11 = (-0.0236809*np.log(T_r)**3 + 0.0049722 *
                   np.log(T_r)**2 + 0.14018*np.log(T_r) - 0.472513)/T_r
    d2_omega_11 = (0.236809*np.log(T_r)**3 - 0.120765*np.log(T_r)
                   ** 2 - 0.0407364*np.log(T_r) + 0.612693)/(T_r**2)
else:
    d2 = -267.00 + (alpha_10*rho_r)**(-2) * (201.57 +
                                             174.672/alpha_10 + (7.3691/alpha_10)**2)
    d4 = 26.7e3 - (alpha_10*rho_r)**(-2) * (19.2265 + 27.6938 /
                                            alpha_10 + (3.29559/alpha_10)**2) * 10**3
    d6 = -8.9e5 + (alpha_10*rho_r)**(-2) * (6.31013 + 10.2266 /
                                            alpha_10 + (2.33033/alpha_10)**2) * 10**5

    X_11 = (rho_r*alpha)**2
    Y_11 = 0.89 + d2/(T_r**2) + d4/(T_r**4) + d6/(T_r**6)
    # First derivative of X and Y
    d1X_11 = (-2*rho_r**2) * ((np.log(V_r) - np.log(T_r)) / T_r)
    d1Y_11 = (-2) * (d2*T_r**4 + 2*d4 * T_r**2 + 3*d6) / (T_r**7)
    # Second derivative of X and Y
    d2X_11 = (-2*rho_r**2) * (np.log(T_r) - (np.log(V_r) - 1) / (T_r**2))
    d2Y_11 = (6*d2*T_r**4 + 20*d4*T_r**2 + 42*d6)/(T_r**8)

    omega_11 = X_11*Y_11
    # First derivative of ln(omega_22)
    d1_omega_11 = d1X_11/X_11 + d1Y_11/Y_11
    # second derivative of ln(omega_22)
    d2_omega_11 = ((d2X_11 - (d1X_11)**2)/X_11 + (d2Y_11 -
                   (d1Y_11)**2)/Y_11) / (d1X_11/X_11 + d1Y_11/Y_11)


# Recursion relations for component 1
A_r = omega_22 / omega_11
E_r = 1 + (T_r/4) * d1_omega_22
C_r = 1 + (T_r/3) * d1_omega_11
# 1 + 3*(C_r - C_r**2) - (T_r**2)*d2_omega_11/3
B_r = 4*C_r - 3*C_r**2 - 1/3*d2_omega_11
H_r = (3*B_r + 6*C_r - 35/4) / (6*C_r - 5)
#F_r = omega_33/omega_11
f_h = 1 + (3/196) * ((8*E_r - 7)**2)
f_D = 1 + (1/8) * (((6*C_r - 5)**2) * (2*A_r + 5)**(-1))
f_l = 1 + (1/42) * ((8*E_r - 7)**2)


# %%
