import pandas as pd
import numpy
from collections import defaultdict, namedtuple, Counter
import warnings
from typing import Optional, List, Union, Dict


# Supported units for temperature
TEMPERATURE_UNITS = ('Celcius')

# Supported units for pressure
PRESSURE_UNITS = ('Pascal')

GasTuple = namedtuple('GasTuple', ['name', 'fraction'])

# Universal constants
h = 6.626176e-34  # Planck constant [J*s]
k_boltzman = 1.380662e-23  # Boltzman constant [J/K]
N_Av = 6.022045e23  # Avogadro constant [1/mol]
R_univ = 8.31441  # Universal gas constant [J/(mol*K)]


# Noble Gas Data

## Helium
He_dict = {
    "Gas" : "He",
    "Molecular Weight" : 4.0026,
    "sigma" : 0.261,
    "eta/k" : 10.4,
    "C*_6" : 3.09,
    "rho" : 0.0797,
    "V*_0" : 850000,
    "Tc" : 5.26,
    "Pc" : 2.26,
    "Vc" : 57,
}


## Neon
Ne_dict = {
    "Gas" : "Ne",
    "Molecular Weight" : 20.179,
    "sigma" : 0.2755,
    "eta/k" : 42,
    "C*_6" : 2.594,
    "rho" : 0.0784,
    "V*_0" : 1109000,
    "Tc" : 44.5,
    "Pc" : 26.9,
    "Vc" : 41.7,
}


## Argon
Ar_dict = {
    "Gas" : "Ar",
    "Molecular Weight" : 39.948,
    "sigma" : 0.335,
    "eta/k" : 141.5,
    "C*_6" : 2.21,
    "rho" : 0.0836,
    "V*_0" : 511700,
    "Tc" : 150.7,
    "Pc" : 48,
    "Vc" : 75.2,
}


## Krypton
Kr_dict = {
    "Gas" : "Kr",
    "Molecular Weight" : 83.8,
    "sigma" : 0.3571,
    "eta/k" : 197.8,
    "C*_6" : 2.164,
    "rho" : 0.0831,
    "V*_0" : 449100,
    "Tc" : 209.4,
    "Pc" : 54.3,
    "Vc" : 92.2,
}

## Xenon
Xe_dict = {
    "Gas" : "Xe",
    "Molecular Weight" : 131.29,
    "sigma" : 0.3885,
    "eta/k" : 274,
    "C*_6" : 2.162,
    "rho" : 0.0854,
    "V*_0" : 38980,
    "Tc" : 289.8,
    "Pc" : 58,
    "Vc" : 118.8,
}


NobleGasData = {
    "He" : He_dict,
    "Ne" : Ne_dict,
    "Ar" : Ar_dict,
    "Kr" : Kr_dict,
    "Xe" : Xe_dict,
}





class NobleGasMix:
    """
    A class to calculate specific properties of noble gas at different temperatures and pressures 
    """
    
        
    def __init__(self, temperature=None, pressure=None):
        # Initialize class attributes
        self._pressure = pressure
        self._pressure_unit = 'Pa'
        self._temperature = temperature
        self._temperature_unit = 'K'
        self._density = None
        self._volume = None
        self._gas_collection = []
        self._molecular_weight = None
        self._molecular_weight_unit = 'g/mol'

    def __repr__(self) -> str:
        string = 'Gas Mixture\n'
        #string += '{: <16}=\t{}'.format('\tDensity', self._density)
        #string += f' [{self._density_units}]\n'
        
        string += '{: <16}=\t{}'.format('\tTemperature', self._temperature)
        string += f' [{self._temperature_unit}]\n'
        

        string += '{: <16}=\t{}'.format('\tPressure', self._pressure)
        string += f' [{self._pressure_unit}]\n'
        
        string += '{: <16}=\t{}'.format('\tMolecular Weight', self._molecular_weight)
        string += f' [{self._molecular_weight_unit}]\n'
        
        string += '{: <16}\n'.format('\tGas Composition')
        for gas, fraction in self._gas_collection:
            string += '{: <16}'.format('\t{}'.format(gas))
            string += f'=\t{fraction: <12}\n'

        return string

    @property
    def gas(self) -> List[namedtuple]:
        return self._gas_collection
    
            
            
            
    
    
    # Universal constants
    h = 6.626176e-34  # Planck constant [J*s]
    k_boltzman = 1.380662e-23  # Boltzman constant [J/K]
    N_Av = 6.022045e23  # Avogadro constant [1/mol]
    R_univ = 8.31441  # Universal gas constant [J/(mol*K)]
    
    
    
    
    def add_gas(self, gas_name: str, Mol_fraction: float):
        """ Add noble gas to mixture
        Parameter:
        ----------
        gas : str
            Nuclide name, limited to: He, Ne, Ar, Kr, Xe
        """
        
        # Check if gas is noble gas
        if gas_name not in ['He', 'Ne', 'Ar', 'Kr', 'Xe']:
            raise ValueError("Not a noble gas, Select from the list: He, Ne, Ar, Kr, Xe'")
        
        self._gas_collection.append(GasTuple(gas_name,Mol_fraction))
        

    # Function to set temperature    
    def set_temperature(self,temp: float, units: Optional[str]=None):
        self._temperature = temp
     
    # Function to set pressure    
    def set_pressure(self,pressure: float, units: Optional[str]=None):
        self._pressure = pressure
    
    # Function to calculate Molecular Weight of the mixture
    def mix_mw(self):
        c1_mf = self._gas_collection[0][1]
        c1_mw = NobleGasData[str(self._gas_collection[0][0])]["Molecular Weight"]
        
        c2_mf = self._gas_collection[1][1]
        c2_mw = NobleGasData[str(self._gas_collection[1][0])]["Molecular Weight"]
        self._molecular_weight = (c1_mf*c1_mw + c2_mf*c2_mw)
    
    
