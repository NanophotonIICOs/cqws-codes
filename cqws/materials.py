"""
This python module sets out classes for calculating the different
properties of common semiconductor materials.


__version__ = '0.1'
__author__ = 'Daniel Martin'
__all__ = ['AlGaAs', 'GaAs', 'AlAs', 'InSb', 'InP', 'InAs', 'GaSb',
           'Custom', 'materials']

"""

import scipy.constants


e0          = scipy.constants.e
hbar        = scipy.constants.hbar
m0          = scipy.constants.electron_mass
meV         = 1e-3;
eV2J        = e0
J2eV        = 1./eV2J

class AlGaAs:
    """
    class object for AlGaAs.
    Properties:
    -----------
    name: string
          Name of class
    x: float
       Proportion of Al in sample (between 0 and 1 inclusive)
    dielectric_constant: float
                         dielectric constant of material (units?)
    me: float
        Effective electron mass (m0)
    mlh: float
         Light hole mass (m0)
    mhh: float
         Heavy hole mass (m0)
    Eg: float
        Band gap energy (eV)
    lattice_constant: float
                      Material lattice constant (A)
    """

    def __init__(self, x, T):
        """
        Sets up properties for AlGaAs.
        Parameters:
        -----------
        x: float
           A float between 0 and 1 inclusive. Determines proportion
           of Aluminium present in semiconductor material.
        T: float
           A float greater than zero. Temperature of material in Kelvin.
        """

        try:
            float(x)
            float(T)
        except TypeError:
            raise TypeError('Amount of Al in AlGaAs and Temperature '
                            'must be a float')
        if (x < 0) or (x > 1):
            raise ValueError('Amount of Al in AlGaAs must be between '
                             '0 and 1')
        if T < 0:
            raise ValueError('Temperature must be at least 0 Kelvin')

        self.name = 'AlGaAs'

        self.x = x
        self.dielectric_constant = 13.18-3.12*x
        self.mlh = (0.082 + 0.068*x)*m0
        self.mhh = (0.51 + 0.25*x)*m0
        self.VBO = 0.67
        if x == 0:
            self.me = 0.063 * m0
            self.Eg = ((1.519 - 5.408e-4*T**2 / (T+204))+1.48*x)
        elif x < 0.45:
            self.me = (0.063 + 0.083*x)*m0
            self.Eg = ((1.519 - 5.408e-4*T**2 / (T+204))+1.48*x)
        else:
            self.me = (0.063 + 0.083*0.45)*m0
            self.Eg = 1.9 + 0.125*x + 0.143*x**2
        self.lattice_constant = 5.6533 + 0.0078*x
        

class GaAs:
    """
    class object for GaAs.
    Properties:
    -----------
    name: string
          Name of class
    dielectric_constant: float
                         dielectric constant of material (units?)
    me: float
        Effective electron mass (m0)
    mlh: float
         Light hole mass (m0)
    mhh: float
         Heavy hole mass (m0)
    Eg: float
        Band gap energy (eV)
    lattice_constant: float
                      Material lattice constant (A)
    """

    def __init__(self, T):
        """
        Sets up properties for GaAs.
        Parameters:
        -----------
        T: float
           A float greater than zero. Temperature of material in Kelvin.
        """

        try:
            float(T)
        except TypeError:
            raise TypeError('Temperature must be a float')
        if T < 0:
            raise ValueError('Temperature must be at least 0 Kelvin')

        self.name = 'GaAs'
        self.dielectric_constant = 13.18
        self.mlh = (0.082)*m0
        self.mhh = (0.51)*m0
        self.me = (0.067)*m0
        self.Eg = 1.519 - 5.408e-4*T**2 / (T+204)
        self.lattice_constant = 5.65325
        self.VBO = 0.67


class AlAs:
    """
    class object for AlAs.
    Properties:
    -----------
    name: string
          Name of class
    dielectric_constant: float
                         dielectric constant of material (units?)
    me: float
        Effective electron mass (m0)
    mlh: float
         Light hole mass (m0)
    mhh: float
         Heavy hole mass (m0)
    Eg: float
        Band gap energy (eV)
    lattice_constant: float
                      Material lattice constant (A)
    """

    def __init__(self, T):
        """
        Sets up properties for AlAs.
        Parameters:
        -----------
        T: float
           A float greater than zero. Temperature of material in Kelvin.
        """

        try:
            float(T)
        except TypeError:
            raise TypeError('Temperature must be a float')
        if T < 0:
            raise ValueError('Temperature must be at least 0 Kelvin')

        self.name = 'AlAs'
        self.dielectric_constant = 10.06
        self.mlh = (0.082 + 0.068)*m0
        self.mhh = (0.51 + 0.25)*m0
        self.me = (0.063 + 0.083*0.45)*m0
        self.Eg = 1.9 + 0.125 + 0.143
        self.lattice_constant = 5.6533 + 0.0078


class InSb:
    """
    class object for InSb.
    Properties:
    -----------
    name: string
          Name of class
    dielectric_constant: float
                         dielectric constant of material (units?)
    me: float
        Effective electron mass (m0)
    mlh: float
         Light hole mass (m0)
    mhh: float
         Heavy hole mass (m0)
    Eg: float
        Band gap energy (eV)
    lattice_constant: float
                      Material lattice constant (A)
    """

    def __init__(self, T):
        """
        Sets up properties for InSb.
        Parameters
        ==========
        T: float
           A float greater than zero. Temperature of material in Kelvin.
        """

        try:
            float(T)
        except TypeError:
            raise TypeError('Temperature must be a float')
        if T < 0:
            raise ValueError('Temperature must be at least 0 Kelvin')

        self.name = 'InSb'
        self.dielectric_constant = 16.8
        self.me = (0.014)*m0
        self.mlh = (0.015)*m0
        self.mhh = (0.43)*m0
        self.Eg = 0.24 - 6e-4 * T**2 / (T + 500)
        self.lattice_constant = 6.479


class InP:
    """
    class object for InP.
    Properties:
    -----------
    name: string
          Name of class
    dielectric_constant: float
                         dielectric constant of material (units?)
    me: float
        Effective electron mass (m0)
    mlh: float
         Light hole mass (m0)
    mhh: float
         Heavy hole mass (m0)
    Eg: float
        Band gap energy (eV)
    lattice_constant: float
                      Material lattice constant (A)
    """

    def __init__(self, T):
        """
        Sets up properties for InP.
        Parameters:
        -----------
        T: float
           A float greater than zero. Temperature of material in Kelvin.
        """

        try:
            float(T)
        except TypeError:
            raise TypeError('Temperature must be a float')
        if T < 0:
            raise ValueError('Temperature must be at least 0 Kelvin')

        self.name = 'InP'
        self.dielectric_constant = 12.5
        self.me = (0.08)*m0
        self.mlh = (0.089)*m0
        self.mhh = (0.6)*m0
        self.Eg = 1.421 - 4.9e-4 * T**2 / (T + 327)
        self.lattice_constant = 5.8687


class InAs:
    """
    class object for InAs.
    Properties:
    -----------
    name: string
          Name of class
    dielectric_constant: float
                         dielectric constant of material (units?)
    me: float
        Effective electron mass (m0)
    mlh: float
         Light hole mass (m0)
    mhh: float
         Heavy hole mass (m0)
    Eg: float
        Band gap energy (eV)
    lattice_constant: float
                      Material lattice constant (A)
    """

    def __init__(self, T):
        """
        Sets up properties for InAs.
        Parameters:
        -----------
        T: float
           A float greater than zero. Temperature of material in Kelvin.
        """

        try:
            float(T)
        except TypeError:
            raise TypeError('Temperature must be a float')
        if T < 0:
            raise ValueError('Temperature must be at least 0 Kelvin')

        self.name = 'InAs'
        self.dielectric_constant = 15.15
        self.me = (0.023)*m0
        self.mlh = (0.026)*m0
        self.mhh = (0.41)*m0
        self.Eg = 0.415 - 2.76e-4 * T**2 / (T + 83)
        self.lattice_constant = 6.0583


class GaSb:
    """
    class object for GaSb.
    Properties:
    -----------
    name: string
          Name of class
    dielectric_constant: float
                         dielectric constant of material (units?)
    me: float
        Effective electron mass (m0)
    mlh: float
         Light hole mass (m0)
    mhh: float
         Heavy hole mass (m0)
    Eg: float
        Band gap energy (eV)
    lattice_constant: float
                      Material lattice constant (A)
    """

    def __init__(self, T):
        """
        Sets up properties for GaSb.
        Parameters:
        -----------
        T: float
           A float greater than zero. Temperature of material in Kelvin.
        """

        try:
            float(T)
        except TypeError:
            raise TypeError('Temperature must be a float')
        if T < 0:
            raise ValueError('Temperature must be at least 0 Kelvin')

        self.name = 'GaSb'
        self.dielectric_constant = 15.7
        self.me = (0.041)*m0
        self.mlh = (0.05)*m0
        self.mhh = (0.4)*m0
        self.Eg = 0.813 - 3.78e-4 * T**2 / (T + 94)
        self.lattice_constant = 6.09593


class Custom:
    """
        class object for Custom material.
        Properties:
        -----------
        name: string
              Name of class
        dielectric_constant: float
                             dielectric constant of material (units?)
        me: float
            Effective electron mass (m0)
        mlh: float
             Light hole mass (m0)
        mhh: float
             Heavy hole mass (m0)
        Eg: float
            Band gap energy (eV)
        lattice_constant: float
                          Material lattice constant (A)
        """

    def __init__(self, Eg, me, mh, mlh, dielectric):
        """
        Sets up properties for Custom material.
        Parameters:
        -----------
        Eg: float
            Band gap energy (eV)
        me: float
            Effective electron mass (m0)
        mh: float
            Heavy hole mass (m0)
        mlh: float
             Light hole mass (m0)
        dielectric: float
                    dielectric constant of material (units?)
        """

        self.name = 'Custom'
        self.dielectric_constant = dielectric
        self.me =  (me)*m0
        self.mlh = (mlh)*m0
        self.mhh = (mh)*m0
        self.Eg = Eg
        self.lattice_constant = 0

materials = {'AlGaAs': AlGaAs, 'AlAs': AlAs, 'GaAs': GaAs,
             'InSb': InSb, 'InP': InP, 'InAs': InAs, 'GaSb': GaSb,
             'Custom': Custom}



    
    
    
    
    
    
    
    
    
    
    
    
    
    