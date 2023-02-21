#import MATERIALS as meterialprop
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import numpy as np
#alen = np.alen
import os
from math import log,exp
from numpy import linalg as LA
from scipy.linalg import eigh 
from numba import jit
#from thomas_solve  import TDMA
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
#import database as database
import scipy.sparse.linalg
from IPython.display import display, Math
from tabulate import tabulate
#from solcore import si, material
from collections import namedtuple
# Time
from datetime import date
from datetime import datetime
from utils import constans, plotstyle.py

today = date.today()

# create data files folder
current_path=os.getcwd()
print("Current PATH:%s"%(current_path))
if os.path.isdir(current_path+'/data'):
    print("The data folder already exist")
else:
    print("The data folder doesn't exist") 
    os.mkdir(current_path+'/data')
    print("The data folder was created successfully!")

save_path=current_path+'/data'


Material = namedtuple('Material',['name','Gap','Me','Mhh','Mlh'])
GaAs = Material('GaAs',lambda T: 1.522 - (5.8E-4*T**2/(300+T)),0.0665,0.55,0.083)
AlAs = Material('AlAs',lambda T: 2.766 - (6e-4*T**2/(408+T)),0.15, 0.81, 0.16)
AlGaAs = Material('AlGaAs',lambda x,T: 1.155*x + 0.37*x**2 - 5.405E-4*T**2/(T+204),
                        lambda x: (AlAs.Me*GaAs.Me)/(x*GaAs.Me + (1-x)*AlAs.Me),
                        lambda x: (AlAs.Mhh*GaAs.Mhh)/(x*GaAs.Mhh + (1-x)*AlAs.Mhh),
                        lambda x: (AlAs.Mlh*GaAs.Mlh)/(x*GaAs.Mlh + (1-x)*AlAs.Mlh))
                
# From Aestimo                
def round2int(x):
    """int is sensitive to floating point numerical errors near whole numbers,
    this moves the discontinuity to the half interval. It is also equivalent
    to the normal rules for rounding positive numbers."""
    # int(x + (x>0) -0.5) # round2int for positive and negative numbers
    return int(x+0.5)

def alen(x):
    return 1 if np.isscalar(x) else len(x)

class Structure(self):
    def __init__(self,T,Fapp,dx,subbands,  #parameters
                  Vc,eps,dop,cb_meff, #arrays
                 **kwargs):

        #value attributes  
        self.T           = T
        self.Fapp        = Fapp
        self.N           = dx        
        self.Vc          = Vc
        self.eps         = eps
        self.dop         = dop
        self.cb_meff     = cb_meff
        self.subbands    = subbands

        # setting any extra parameters provided with initialisation
        for key,value in kwargs.items():
            setattr(self,key,value)
        

class AttrDict(dict):
    """turns a dictionary into an object with attribute style lookups"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    
class StructureFrom(Structure):
    def __init__(self,inputfile):
        if type(inputfile)==dict:
            inputfile=AttrDict(inputfile)            
        # Parameters for simulation
        self.structure_name = inputfile.structure_name  
        self.Fapp           = inputfile.Fapp
        self.T              = inputfile.T
        self.dx             = inputfile.gridfactor*1e-9 #grid in m
        self.subbands       =  inputfile.subbands
        self.HHBinding      = inputfile.HHBinding
        self.LHBinding      = inputfile.LHBinding
        self.Qc             = inputfile.Qc
        self.Qv             = inputfile.Qv
        # Loading material list
        self.material = inputfile.material
        totallayer = alen(self.material)
        print("Total layer number: %d" %(totallayer))
        # Calculate the required number of grid points
        self.x_max = sum([layer[0] for layer in self.material])*1e-9 #total thickness (m)
        self.n_max = int(self.x_max/self.dx)
        self.create_structure_arrays()
        self.xaxis    = np.arange(0,self.n_max)*self.dx
        
        
        
    def create_structure_arrays(self):
        """ initialise arrays/lists for structure"""
        # Calculate the required number of grid points
        self.x_max = sum([layer[0] for layer in self.material])*1e-9 #total thickness (m)
        n_max      = round2int(self.x_max/self.dx)
        self.n_max = n_max
        dx         = self.dx
        #material_property = self.material_property
        #alloy_property    = self.alloy_property
        cb_meff           = np.zeros(n_max)     #conduction band effective mass to electrons
        vblh_meff         = np.zeros(n_max)     #valence band effective mass to holes
        vbhh_meff         = np.zeros(n_max)     #valence band effective mass to holes
        cb                = np.zeros(n_max) 
        vb                = np.zeros(n_max) 
        Eg                = np.zeros(n_max)            #bandgap energy (?)
        eps               = np.zeros(n_max)        #dielectric constant
        self.dop           = np.zeros(n_max)           #doping
        T                 = self.T
        position          = 0.0 # keeping in nanometres (to minimise errors)
        
        # Donmez et al. Nanoscale Research Letters 2012 7:622
        # Functions to calculate Gap as a function of temperature and composition
        EgGaAs   = lambda T: 1.519 - (5.405E-4*T**2/(204+T))
        EgInAs   = lambda T: 0.415 - (2.76E-4*T**2/(83+T))
        #EgGaAs   = lambda T: 1.5216 - (8.871E-4*T**2/(572+T))
        #EgGaAs   = lambda T: 1.519 - (5.6E-4*T**2/(226+T))
        EgAlGaAs = lambda x,T: 1.155*x + 0.37*x**2 - 5.405E-4*T**2/(T+204)
        #EgAlGaAs = lambda x,T : 1.247*x - 0.0005405*T**2/(T+204)
        EgAlAs   = lambda T: 2.239 - (6.0e-4*T**2/(408+T))
        #Functions to calculate Effective Mass as a function of composition
        meGaAs    = 0.0665
        mlhGaAs   = 0.094
        mhhGaAs   = 0.34
        meAlAs    = 0.154
        mlhAlAs   = 0.16
        mhhAlAs   = 0.76
        meInAs    = 0.023
        mhhInAs   =  0.41
        mlhInAs   =  0.026
        meAlGaAs  = lambda x: (meAlAs*meGaAs)/(x*meGaAs + (1-x)*meAlAs)
        mlhAlGaAs = lambda x: (mlhAlAs*mlhGaAs)/(x*mlhGaAs + (1-x)*mlhAlAs)
        mhhAlGaAs = lambda x: (mhhAlAs*mhhGaAs)/(x*mhhGaAs + (1-x)*mhhAlAs)
        
        # Band Offset ratios
        Qc = self.Qc
        Qv = self.Qv
        
        
        
        for layer in self.material:
            startindex  = round2int(position*1e-9/dx)
            position   += layer[0] # update position to end of the layer
            finishindex = round2int(position*1e-9/dx)
            matType     = layer[1]  
            x           = layer[2]

            if matType == 'GaAs':
                cb[startindex:finishindex] =  (EgGaAs(T))*q #material("GaAs")(T=self.T).band_gap
                vb[startindex:finishindex] = 0
                cb_meff[startindex:finishindex]   = meGaAs*m_e
                vblh_meff[startindex:finishindex] = mlhGaAs*m_e
                vbhh_meff[startindex:finishindex] = mhhGaAs*m_e
            
            elif matType == 'AlAs':
                cb[startindex:finishindex] =  (EgGaAs(T)+EgAlAs(T)*Qc)*q
                vb[startindex:finishindex] = -(EgAlAs(T)*Qv)*q
                cb_meff[startindex:finishindex]  =  meAlAs*m_e
                vblh_meff[startindex:finishindex] = mlhAlAs*m_e
                vbhh_meff[startindex:finishindex] = mhhAlAs*m_e
            elif matType == 'InAs':
                cb[startindex:finishindex] =  (EgGaAs(T)-EgInAs(T)*Qc)*q
                vb[startindex:finishindex] = -(EgInAs(T)*Qv)*q
                cb_meff[startindex:finishindex]  =  meInAs*m_e
                vblh_meff[startindex:finishindex] = mlhInAs*m_e
                vbhh_meff[startindex:finishindex] = mhhInAs*m_e
        
            elif matType == 'AlGaAs':
                cb[startindex:finishindex] =  (EgGaAs(T)+EgAlGaAs(x,T)*Qc)*q
                vb[startindex:finishindex] =  -(EgAlGaAs(x,T)*Qv)*q
                cb_meff[startindex:finishindex]   = meAlGaAs(x)*m_e
                vblh_meff[startindex:finishindex] = mlhAlGaAs(x)*m_e
                vbhh_meff[startindex:finishindex] = mhhAlGaAs(x)*m_e

            elif matType == 'Vacuum':
                cb[startindex:finishindex] = (EgGaAs(T)+0.5)*q
                vb[startindex:finishindex] =  -(EgGaAs(T)+0.5)*q
                cb_meff[startindex:finishindex]   = m_e
                vblh_meff[startindex:finishindex] = m_e
                vbhh_meff[startindex:finishindex] = m_e
                
                
            if layer[4] == 'n':  
                chargedensity = layer[3]*1e6 #charge density in m**-3 (conversion from cm**-3)
            elif layer[4] == 'p': 
                chargedensity = -layer[3]*1e6 #charge density in m**-3 (conversion from cm**-3)
            else:
                chargedensity = 0.0
            
            self.dop[startindex:finishindex] = chargedensity
        
        self.count_layer=0
        for layer in self.material:
            if layer[5]=='well':
                self.count_layer+=layer[0]
            elif layer[5]=='cbarrier':
                self.count_layer+=layer[0]


        
        self.cb         = cb
        self.vb         = vb
        self.cb_meff    = cb_meff
        self.vblh_meff  = vblh_meff
        self.vbhh_meff  = vbhh_meff

        
            
#--------------------------------- Schrodinger SOLVER 
def H(vpot,mass,Fapp,dx,n):
    vb   = vpot 
    meff = mass
    H    = np.zeros((n,n))
    m_minus  = (meff[0] + meff[1])/2.0
    m_plus   = (meff[0] + meff[1]) /2.0
    sn_minus =  -pow(hbar/dx,2)/(2.0*m_minus)
    sn_plus  =  -pow(hbar/dx,2)/(2.0*m_plus)
    bi       =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    H[0,0]  = bi + vb[0]
    H[0,1]  = sn_plus
    for ii in range (1,n-1):
        m_minus      = (meff[ii] + meff[ii-1])/2.0
        m_plus       = (meff[ii+1]    + meff[ii])/2.0
        sn_minus     = -pow(hbar/dx,2)/(2.0*m_minus)
        sn_plus      = -pow(hbar/dx,2)/(2.0*m_plus)
        bi           =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
        H[ii,ii-1]  = sn_minus
        H[ii,ii]    = bi + vb[ii] 
        H[ii,ii+1]  = sn_plus
    m_minus     = (meff[n-2] + meff[n-1])/2.0
    m_plus      = (meff[n-1] +  meff[n-2])/2.0
    sn_minus    =  -pow(hbar/dx,2)/(2*m_minus)
    sn_plus     =  -pow(hbar/dx,2)/(2*m_plus)
    bi          =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    H[n-1,n-1]  = bi + vb[n-1] 
    H[n-1,n-2]  = sn_minus
    return H

def Hholes(vpot,mass,Fapp,dx,n):
    vb   = vpot 
    meff = mass
    H    = np.zeros((n,n))
    m_minus  = (meff[0] + meff[1])/2.0
    m_plus   = (meff[0] + meff[1]) /2.0
    sn_minus =  -pow(hbar/dx,2)/(2.0*m_minus)
    sn_plus  =  -pow(hbar/dx,2)/(2.0*m_plus)
    bi       =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    H[0,0]  = bi - vb[0]
    H[0,1]  = sn_plus
    for ii in range (1,n-1):
        m_minus      = (meff[ii] + meff[ii-1])/2.0
        m_plus       = (meff[ii+1]    + meff[ii])/2.0
        sn_minus     = -pow(hbar/dx,2)/(2.0*m_minus)
        sn_plus      = -pow(hbar/dx,2)/(2.0*m_plus)
        bi           =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
        H[ii,ii-1]  = sn_minus
        H[ii,ii]    = bi - vb[ii] 
        H[ii,ii+1]  = sn_plus
    m_minus     = (meff[n-2] + meff[n-1])/2.0
    m_plus      = (meff[n-1] +  meff[n-2])/2.0
    sn_minus    =  -pow(hbar/dx,2)/(2*m_minus)
    sn_plus     =  -pow(hbar/dx,2)/(2*m_plus)
    bi          =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    H[n-1,n-1]  = bi - vb[n-1] 
    H[n-1,n-2]  = sn_minus
    return H



            
def Schrodinger(model,sparse = False,absolute = False):
    dx       = model.dx
    n        = model.n_max
    subbands = model.subbands 
    xaxis    = np.arange(0,n)*dx 
    Fapp     = model.Fapp
    pote     = model.cb + Fapp*q*xaxis
    poth     = model.vb + Fapp*q*xaxis
    me       = model.cb_meff
    HHBinding = model.HHBinding
    LHBinding = model.LHBinding
    mlh      = model.vblh_meff
    mhh      = model.vbhh_meff
    
    Ee    = np.zeros(subbands)
    Elh   = np.zeros(subbands)
    Ehh   = np.zeros(subbands)
    psie  = np.zeros((subbands,n))
    psihh = np.zeros((subbands,n))
    psilh = np.zeros((subbands,n))
    EHH   = np.zeros(subbands)
    ELH   = np.zeros(subbands)
    Hamiltonian_e  = H(pote,me,Fapp,dx,n)
    Hamiltonian_lh = H(-poth,mlh,Fapp,dx,n)
    Hamiltonian_hh = H(-poth,mhh,Fapp,dx,n)
    meV = 1e3
    
    Pcb = pote
    Pvb = poth
    
    
    if sparse == False:
        Ene,WFe   = eigh(Hamiltonian_e)
        Enhh,WFhh = eigh(Hamiltonian_hh)
        Enlh,WFlh = eigh(Hamiltonian_lh)
        for i in range(subbands):
            Ee[i]    = Ene[i]/q  # eV
            Elh[i]   = Enlh[i]/q 
            Ehh[i]   = Enhh[i]/q
            potcb    = Pcb[:]/q
            potvb    = Pvb[:]/q
    
        # Export Wavefunctions
        if absolute == False:
            for i in range(subbands):
                psie[i]  = WFe[:,i]
                psihh[i] = WFhh[:,i]
                psilh[i] = WFlh[:,i]
        else:
            for i in range(subbands):
                psie[i]  = np.power(np.absolute(WFe[:,i]),2)
                psihh[i] = np.power(np.absolute(WFhh[:,i]),2)
                psilh[i] = np.power(np.absolute(WFlh[:,i]),2)
            
    else:
        #To Electrons
        upper_e     = np.diag(Hamiltonian_e,k=1)
        diagonal_e  = np.diag(Hamiltonian_e,k=0)
        lower_e     = np.diag(Hamiltonian_e,k=-1)
        H_diags_e = [lower_e,diagonal_e,upper_e]
        Hn_e = scipy.sparse.diags(H_diags_e, [-1,0,1], format='csc')
        energy_e,wfe_e = eigsh(Hn_e, k= model.subbands, sigma = 0.0,which = 'LM')
        energy_e = energy_e.real
        wfe_e = wfe_e
        #To Light Holes
        upper_lh     = np.diag(Hamiltonian_lh,k=1)
        diagonal_lh  = np.diag(Hamiltonian_lh,k=0)
        lower_lh     = np.diag(Hamiltonian_lh,k=-1)
        H_diags_lh = [lower_lh,diagonal_lh,upper_lh]
        Hn_lh = scipy.sparse.diags(H_diags_lh, [-1,0,1], format='csc')
        energy_lh,wfe_lh = eigsh(Hn_lh, k= model.subbands, sigma = 0.0,which = 'LM')
        energy_lh = energy_lh.real
        wfe_lh = wfe_lh
        #To Heavy Holes
        upper_hh     = np.diag(Hamiltonian_hh,k=1)
        diagonal_hh  = np.diag(Hamiltonian_hh,k=0)
        lower_hh     = np.diag(Hamiltonian_hh,k=-1)
        H_diags_hh   = [lower_hh,diagonal_hh,upper_hh]
        Hn_hh        = scipy.sparse.diags(H_diags_hh, [-1,0,1], format='csc')
        energy_hh,wfe_hh = eigsh(Hn_hh, k= model.subbands, sigma = 0.0,which = 'LM')
        energy_hh = energy_hh.real
        wfe_hh = wfe_hh
        
        if absolute == False:
            for i in range(subbands):
                psie[i]  = wfe_e[:,i]
                psilh[i] = wfe_lh[:,i]
                psihh[i] = wfe_hh[:,i]
        else:
            for i in range(subbands):
                psie[i]  = np.power(np.absolute(wfe_e[:,i]),2)
                psilh[i] = np.power(np.absolute(wfe_lh[:,i]),2)
                psihh[i] = np.power(np.absolute(wfe_hh[:,i]),2)
            
        for i in range(subbands):
            Ee[i] = energy_e[i]/q  # eV
            Elh[i] = energy_lh[i]/q
            Ehh[i] = energy_hh[i]/q
            potcb  = Pcb/q
            potvb  = Pvb/q

        
    for i in range(subbands):
        EHH[i] = Ee[i]+Ehh[i]-HHBinding
        ELH[i] = Ee[i]+Elh[i]-LHBinding
        display(Math(r'\text{Transition} \,E_{%d}-HH_{%d}: %.4f'%(i,i,EHH[i])))
        display(Math(r'\text{Transition} \,E_{%d}-LH_{%d}: %.4f'%(i,i,ELH[i]))) 

        
        
        
        
        
        
        
    class Results(): pass
    results          = Results()
    results.xaxis    = xaxis
    results.psie     = psie
    results.Ee       = Ee
    results.psilh    = psilh
    results.Elh      = Elh
    results.psihh    = psihh
    results.Ehh      = Ehh
    results.cb       = potcb
    results.vb       = potvb
    results.dx       = dx
    results.subbands = subbands
    results.TEHH     = EHH
    results.TELH     = ELH
    return results







class Solver:   
    def __init__(self,model):
        self.dx       = model.dx
        self.n        = model.n_max
        self.subbands = model.subbands 
        self.xaxis    = np.arange(0,self.n)*self.dx 
        self.Fapp     = model.Fapp
        self.pote     = model.cb - self.Fapp*q*self.xaxis
        self.poth     = model.vb - self.Fapp*q*self.xaxis 
        self.me       = model.cb_meff
        self.mlh      = model.vblh_meff
        self.mhh      = model.vbhh_meff
        self.HHBinding   = model.HHBinding
        self.LHBinding   = model.LHBinding
        self.T           = model.T
        self.Qc          = model.Qc
        self.Qv          = model.Qv
        self.material  = model.material
        self.structure_name=model.structure_name
        
    
    def QuantumSolutions(self,absolute = False,Print = False):
            self.Ee    = np.zeros(self.subbands)
            self.Elh   = np.zeros(self.subbands)
            self.Ehh   = np.zeros(self.subbands)
            self.psie  = np.zeros((self.n,self.subbands))
            self.psihh = np.zeros((self.n,self.subbands))
            self.psilh = np.zeros((self.n,self.subbands))
            self.EHH   = np.zeros(self.subbands)
            self.ELH   = np.zeros(self.subbands)
            self.Hamiltonian_e  = H(self.pote,self.me,self.Fapp,self.dx,self.n)
            self.Hamiltonian_lh = Hholes(self.poth,self.mlh,self.Fapp,self.dx,self.n)
            self.Hamiltonian_hh = Hholes(self.poth,self.mhh,self.Fapp,self.dx,self.n)
            meV = 1e3
    
            self.Pcb = self.pote
            self.Pvb = self.poth
            
            # Eigenvalues and Eigenvectors to electrons in Conduction bands
            upper_e       = np.diag(self.Hamiltonian_e,k=1)
            diagonal_e    = np.diag(self.Hamiltonian_e,k=0)
            lower_e       = np.diag(self.Hamiltonian_e,k=-1)
            H_diags_e     = [lower_e,diagonal_e,upper_e]
            Hn_e          = scipy.sparse.diags(H_diags_e, [-1,0,1], format='csc')
            self.energy_e,self.wfe_e = eigsh(Hn_e, k=self.subbands, sigma = 0.0,which = 'LM')
            self.energy_e = self.energy_e.real
            self.wfe_e    = self.wfe_e.real
            
            #To Light Holes
            upper_lh       = np.diag(self.Hamiltonian_lh,k=1)
            diagonal_lh    = np.diag(self.Hamiltonian_lh,k=0)
            lower_lh       = np.diag(self.Hamiltonian_lh,k=-1)
            H_diags_lh     = [lower_lh,diagonal_lh,upper_lh]
            Hn_lh          = scipy.sparse.diags(H_diags_lh, [-1,0,1], format='csc')
            self.energy_lh,self.wfe_lh = eigsh(Hn_lh, k=self.subbands, sigma = 0.0,which = 'LM')
            self.energy_lh = self.energy_lh.real
            self.wfe_lh    = self.wfe_lh.real
            
            
            #To Heavy Holes
            upper_hh       = np.diag(self.Hamiltonian_hh,k=1)
            diagonal_hh    = np.diag(self.Hamiltonian_hh,k=0)
            lower_hh       = np.diag(self.Hamiltonian_hh,k=-1)
            H_diags_hh     = [lower_hh,diagonal_hh,upper_hh]
            Hn_hh          = scipy.sparse.diags(H_diags_hh, [-1,0,1], format='csc')
            self.energy_hh,self.wfe_hh = eigsh(Hn_hh, k=self.subbands, sigma = 0.0,which = 'LM')
            self.energy_hh = self.energy_hh.real
            self.wfe_hh    = self.wfe_hh.real
        
        
            if absolute == False:
                for i in range(self.subbands):
                    self.psie[:,i]  = self.wfe_e[:,i]
                    self.psilh[:,i] = self.wfe_lh[:,i]
                    self.psihh[:,i] = self.wfe_hh[:,i]
            else:
                for i in range(self.subbands):
#                     self.psie[:,i]  = np.power(np.absolute(self.wfe_e[:,i]),2)
#                     self.psilh[:,i] = np.power(np.absolute(self.wfe_lh[:,i]),2)
#                     self.psihh[:,i] = np.power(np.absolute(self.wfe_hh[:,i]),2)
                      self.psie[:,i]  = self.wfe_e[:,i]*self.wfe_e[:,i]
                      self.psilh[:,i] = self.wfe_lh[:,i]*self.wfe_lh[:,i]
                      self.psihh[:,i] = self.wfe_hh[:,i]*self.wfe_hh[:,i]
            
            for i in range(self.subbands):
                self.Ee[i]  = self.energy_e[i]/q  # eV
                self.Elh[i] = self.energy_lh[i]/q
                self.Ehh[i] = self.energy_hh[i]/q
                self.potcb  = self.Pcb/q
                self.potvb  = self.Pvb/q

        
            for i in range(self.subbands):
                self.EHH[i] = self.Ee[i]+self.Ehh[i]-self.HHBinding
                self.ELH[i] = self.Ee[i]+self.Elh[i]-self.LHBinding
            
            self.DirectTransitions = []
            for i in range(self.subbands):
                if i%2 == 0:
                    self.DirectTransitions.append(["E%d->HH%d :" %(i+1,i+1), " %.4f"%(self.EHH[i])])
                    self.DirectTransitions.append(["E%d->LH%d :" %(i+1,i+1), " %.4f"%(self.ELH[i])])
                else:
                     self.DirectTransitions.append(["E%d->HH%d :" %(i+1,i+1), " %.4f"%(self.EHH[i])])
                     self.DirectTransitions.append(["E%d->LH%d :" %(i+1,i+1), " %.4f"%(self.ELH[i])])

                
        
            
            #self.TotalT = np.vstack((self.EHH,self.ELH))
            self.energies = np.vstack((self.Ee,self.Ehh,self.Elh))      
            self.eprinte = np.zeros((self.subbands,3),dtype = object)
            for ii in range(self.subbands):
                self.eprinte[ii,:] = [str('E'+str(ii+1)+'-> '+'%.4f'%self.Ee[ii]),
                                      str('HH'+str(ii+1)+'->'+'%.4f'%-self.Ehh[ii]),
                                      str('LH'+str(ii+1)+'->'+'%.4f'%-self.Elh[ii])]
           
                                     
            
            if Print == True:
                print('Direct Transitions')
                for i in range(self.subbands):
                    display(Math(r'\text{Transition} \,E_{%d}-HH_{%d}: %.4f'%(i+1,i+1,self.EHH[i])))
                    display(Math(r'\text{Transition} \,E_{%d}-LH_{%d}: %.4f'%(i+1,i+1,self.ELH[i]))) 
                print(tabulate(self.eprinte,['ELECTRON [eV]','HEAVY HOLE [eV]','LIGHT HOLE [eV]'],
                              tablefmt='orgtbl',stralign='center',floatfmt='.4f'))

            
            
            class Results(): pass
            results          = Results()
            results.xaxis    = self.xaxis
            results.psie     = self.psie
            results.Ee       = self.Ee
            results.psilh    = self.psilh
            results.Elh      = self.Elh
            results.psihh    = self.psihh
            results.Ehh      = self.Ehh
            results.cb       = self.potcb
            results.vb       = self.potvb
            results.dx       = self.dx
            results.subbands = self.subbands
            results.TEHH     = self.EHH
            results.TELH     = self.ELH
            results.Energies = self.energies
            results.PrintEn  = self.eprinte
            results.DirectTransitions = self.DirectTransitions
            return results
        
    def print_result(self,structure_name,results):
        self.eprinte = results.PrintEn
        trans_file = open('Energies-of-sample-%s.txt'%structure_name,'w')
        trans_file.write('PROGRAM STARTS ON %s' %today+'  AT %s\n' %datetime.time(datetime.now()))
        trans_file.write('THIS PROGRAM CALCULATE OPTICAL TRANSITIONS OF SAMPLE: '+structure_name+'\n\n')
        trans_file.write(tabulate([["Parameters"],
                                    ["Temperature:         %.1f"%self.T],
                                    ["Band Offset Ratio:   %.2f:%.2f"%(self.Qc,self.Qv)],
                                    ["Step size (z):       %.3f nm"%(results.dx/1e-9)],
                                    ["Heavy Hole Binding:  %.2f  meV"%(self.HHBinding*1000)],
                                    ["Light Hole Binding:  %.2f  meV"%(self.LHBinding*1000)]],
                                     headers="firstrow"))
        trans_file.write('\n\n')
        trans_file.write('THE SIMPLE STRUCTURE OF THIS SAMPLE:\n\n')
        trans_file.write(tabulate(self.material,headers=["Width","Material", "x Composition","Dopped cm^-3","Dopped type", 'Type']))
        trans_file.write('\n\nENERGIES:\n\n')
        trans_file.write(tabulate(self.eprinte,['ELECTRON [eV]','HEAVY HOLE [eV]','LIGHT HOLE[eV]'],tablefmt='orgtbl',stralign='center',floatfmt='.4f'))
        trans_file.write('\n\nTRANSITIONS\n\n')
        trans_file.write(tabulate(results.DirectTransitions,headers=["Transition","Value (eV)"],tablefmt="presto"))
        trans_file.write('\n')
        trans_file.close()
        
        
    def plotting(self,results,amp=10,axmin=20,axmax=20,eymin = 0,eymax=0,hymin=0,hymax=0,save=False):
            # Data
            self.subbands = results.subbands
            self.cb   = results.cb
            self.vb   = results.vb
            self.WF_e = results.psie
            self.WF_hh= results.psihh
            self.WF_lh= results.psilh
            self.Ee   = results.Ee
            self.Ehh  = results.Ehh
            self.Elh  = results.Elh
        
            colors = ['b','r','g','orange','purple']
            
            xmin = ((self.xaxis[self.n-1]/nm)/2)-axmin
            xmax = ((self.xaxis[self.n-1]/nm)/2)+axmax
            eymin  = min(self.cb) + eymin
            eymax  = max(self.cb) + eymax
            hymin  = min(self.vb) + hymin
            hymax  = max(self.vb) + hymax
            
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(5,7))
            f.subplots_adjust(hspace=0.05)
            # Plot electrons and heavy holes
            ax1.plot(self.xaxis/nm,self.cb,ls='-',lw='2',color='gray')
            for i in range(self.subbands):
                ax1.plot(self.xaxis/nm,amp*self.WF_e[:,i]+self.Ee[i],
                        ls='-',
                        lw='2',
                        color=colors[i],
                        label = '$\psi e_%d$'%(i))
            
            ax2.plot(self.xaxis/nm,self.vb,ls='-',lw='2',color='gray')
            
            for i in range(self.subbands):
                ax2.plot(self.xaxis/nm,amp*self.WF_hh[:,i]-self.Ehh[i],
                         ls='-',
                         lw='2',
                         color=colors[i],
                         label = '$\psi hh_%d$'%(i))
                ax2.plot(self.xaxis/nm,amp*self.WF_e[:,i]-self.Elh[i],
                        ls=':',
                        lw='2',
                        color=colors[i],
                        label = '$\psi e_%d$'%(i))
                #ax2.plot(self.xaxis/nm,self.WF_lh[:,i]-self.Elh[i],ls='-',lw='2')
            ax1.legend(loc = 2,fontsize=10,frameon=False)
            ax1.set_ylabel(r'CB-edge (eV)',fontsize=20)
            ax1.yaxis.set_label_coords(-0.14,0.5)
            ax1.set_xlim(xmin,xmax)  
            #ax1.set_ylim(eymin,eymax)
            # outliers only
            ax2.legend(loc = 2,fontsize=10,frameon=False)
            #ax2.set_ylim(hymin,hymax)  # most of the data
            ax2.set_ylabel(r'VB-edge (eV)',fontsize=17)
            ax2.set_xlabel(r'$\mathrm{Growth\,\, Direction\,\, [nm]}$',fontsize=17)
            ax2.yaxis.set_label_coords(-0.14,0.5)
            
            # hide the spines between ax and ax2
            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.tick_params(
                             axis='x',          # changes apply to the x-axis
                             which='both',      # both major and minor ticks are affected
                             bottom=False,      # ticks along the bottom edge are off
                             top=False,         # ticks along the top edge are off
                             labelbottom=False)
            ax2.xaxis.tick_bottom()
            
            d = .05
            l = 1
            # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=ax1.transAxes,lw=2, color='k', clip_on=False)
            ax1.plot((0, +d), (0, 0), **kwargs)        # top-left diagonal
            ax1.plot((l - d, 0 + l), (0, +0), **kwargs)


            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((0, +d), (1 , 1 + 0), **kwargs)  # bottom-left diagonal
            ax2.plot((1 - d, 1 + 0), (1 - 0, 1 + 0), **kwargs)  # bottom-right diagonal
            
            if save==True:
                if os.path.isdir(current_path+'/plots'):
                    print("The plots folder already exist")
                else:
                    print("The plots folder doesn't exist") 
                    os.mkdir(current_path+'/plots')
                    print("The plots folder was created successfully!")

                plt.savefig('plots/%s.png'%(self.structure_name),dpi=300,bbox_inches='tight',transparent=True)
            plt.show()


    def save_data(self,results,**kwargs):
        nx,ny =results.psie.shape
        bands = np.zeros((nx,3))
        psie = np.zeros((nx,ny+1))
        psihh = np.zeros((nx,ny+1))
        psilh = np.zeros((nx,ny+1))
        bands[:,0]=results.xaxis/nm
        psie[:,0]=results.xaxis/nm
        psihh[:,0]=results.xaxis/nm
        psilh[:,0]=results.xaxis/nm
        
        bands[:,1]=results.cb
        bands[:,2]=results.vb

        for key, value in kwargs.items():
            if key=='absolute':
                abs_value=value
    
       
        for i in range(results.subbands):
            psie[:,1+i] =results.psie[:,i]  + results.Ee[i]
            psilh[:,1+i]=results.psilh[:,i] - results.Ehh[i]
            psihh[:,1+i]=results.psihh[:,i] - results.Elh[i]

        np.savetxt(save_path+'/'+self.structure_name+'-band-edge.txt',bands,delimiter=',')
        np.savetxt(save_path+'/'+self.structure_name+'-wf-e.txt',psie,delimiter=',')
        np.savetxt(save_path+'/'+self.structure_name+'-wf-hh.txt',psihh,delimiter=',')
        np.savetxt(save_path+'/'+self.structure_name+'-wf-lh.txt',psilh,delimiter=',')

      
    def plotting_subbands(self,results,symmetric = False,amp=10,axmin=0,axmax=1,eymin = 1.515,eymax=2,hymin=0,hymax=-0.1):
            plt.rcParams['xtick.labelsize']     = 13
            plt.rcParams['ytick.labelsize']     = 13
            plt.rcParams['axes.linewidth']      = 1
            plt.rcParams["xtick.minor.visible"] =  True
            plt.rcParams["xtick.major.size"]    =  8
            plt.rcParams["xtick.minor.size"]    =  4
            plt.rcParams["xtick.major.width"]   =  1
            plt.rcParams["xtick.minor.width"]   =  1
            plt.rcParams["xtick.direction"]     =  'in'
            plt.rcParams["ytick.minor.visible"] =  True
            plt.rcParams["ytick.major.size"]    =  8
            plt.rcParams["ytick.minor.size"]    =  4
            plt.rcParams["ytick.major.width"]   =  1
            plt.rcParams["ytick.minor.width"]   =  1
            plt.rcParams["ytick.direction"]     =  'in'
            #plt.rcParams['text.usetex']         = True
            plt.rcParams['legend.frameon']      = False
            nm = 1e-9

            # Data
            self.subbands = results.subbands
            self.cb   = results.cb
            self.vb   = results.vb
            self.WF_e = results.psie
            self.WF_hh= results.psihh
            self.WF_lh= results.psilh
            self.Ee   = results.Ee
            self.Ehh  = results.Ehh
            self.Elh  = results.Elh
            
            
            colors = ['b','r','g','orange','purple']
            
            f, (ax11, ax22) = plt.subplots(2, 1, sharex=True,figsize=(3,5))
            for i in range(self.subbands):
            
                ax11.plot([0.2,0.8],[self.Ee[i],self.Ee[i]],'-b')
            
            
            
            
            ax11.set_ylim(eymin,eymax)
            ax11.set_xlim(axmin,axmax) 
            
            
             # hide the spines between ax and ax2
            ax11.spines['bottom'].set_visible(False)
            ax22.spines['top'].set_visible(False)
            ax11.xaxis.tick_top()
            ax11.tick_params(labeltop=False)  # don't put tick labels at the top
            ax22.xaxis.tick_bottom()
            
            
                
            d = .01  
            l = 1
            # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=ax11.transAxes, lw=1,color='k', clip_on=False)
            ax11.plot((0, +d), (0, 0), **kwargs)        # top-left diagonal
            ax11.plot((l - d, 0 + l), (0, +0), **kwargs)


            kwargs.update(transform=ax22.transAxes)  # switch to the bottom axes
            ax22.plot((0, +d), (1 , 1 + 0), **kwargs)  # bottom-left diagonal
            ax22.plot((1 - d, 1 + 0), (1 - 0, 1 + 0), **kwargs)  # bottom-right diagonal
            
            plt.show()
            

            
            

        
        
        
