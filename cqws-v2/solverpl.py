import MATERIALS as meterialprop
import matplotlib.pyplot as pl
import numpy as np
alen = np.alen
import os
from math import log,exp
from numpy import linalg as LA
from scipy.linalg import eigh 
from numba import jit
from thomas_solve  import TDMA
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigs
import database as database
import scipy.sparse.linalg
from IPython.display import display, Math
import math
#Defining constants and material parameters
q    = 1.602176e-19 #C
kBe  = 8.6173303e-5 # eV / K
kb   = 1.3806504e-23 #J/K
kbe  = 8.6173324e-5 #eV/K
nii  = 0.0
hbar = 1.054588757e-34
m_e  = 9.1093826E-31 #kg
pi   = np.pi
eps0 = 8.8541878176e-12 #F/m

J2eV=1/q #Joules to eV
eV2J=1*q #eV to Joules



# From Aestimo



def round2int(x):
    """int is sensitive to floating point numerical errors near whole numbers,
    this moves the discontinuity to the half interval. It is also equivalent
    to the normal rules for rounding positive numbers."""
    # int(x + (x>0) -0.5) # round2int for positive and negative numbers
    return int(x+0.5)



class Structure():
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
        self.scheme      = scheme
        self.Binding     = Binding
        self.Emin        = Emin
        self.Emax        = Emax
        self.delta       = delta
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
            inputfile=AttrDict(inputfile,database)            
        # Parameters for simulation
        self.Fapp     = inputfile.Fapp
        self.T        = inputfile.T
        self.dx       = inputfile.gridfactor*1e-9 #grid in m
        self.subbands =  inputfile.subbands
        self.Binding  = inputfile.Binding 
        self.Emin     = inputfile.Emin
        self.Emax     = inputfile.Emax
        self.delta    = inputfile.delta
        # Loading material list
        self.material = inputfile.material
        totallayer = alen(self.material)
        print("Total layer number: %d" %(totallayer))
        self.material_property = database.materialproperty
        self.alloy_property = database.alloyproperty
        # Calculate the required number of grid points
        self.x_max = sum([layer[0] for layer in self.material])*1e-9 #total thickness (m)
        self.n_max = int(self.x_max/self.dx)
                
        #cb_meff #conduction band effective mass (array, len n_max)
        #fi #Bandstructure potential (array, len n_max)
        #eps #dielectric constant (array, len n_max)
        #dop #doping distribution (array, len n_max)
        self.create_structure_arrays()
        
    def create_structure_arrays(self):
        """ initialise arrays/lists for structure"""
        # Calculate the required number of grid points
        self.x_max = sum([layer[0] for layer in self.material])*1e-9 #total thickness (m)
        n_max      = round2int(self.x_max/self.dx)
        self.n_max = n_max
        dx         = self.dx
        material_property = self.material_property
        alloy_property    = self.alloy_property
        cb_meff           = np.zeros(n_max)     #conduction band effective mass to electrons
        vblh_meff         = np.zeros(n_max)     #valence band effective mass to holes
        vbhh_meff         = np.zeros(n_max)     #valence band effective mass to holes
        CB                = np.zeros(n_max) 
        VB                = np.zeros(n_max) 
        Eg                = np.zeros(n_max)            #bandgap energy (?)
        eps               = np.zeros(n_max)        #dielectric constant
        dop               = np.zeros(n_max)           #doping
        T                 = self.T
        position          = 0.0 # keeping in nanometres (to minimise errors)
        
        # Donmez et al. Nanoscale Research Letters 2012 7:622
        # Functions to calculate Gap as a function of temperature and composition
        aplha    = 5.405e-4
        beta     = 204
        EgGaAs   = lambda T: 1.519 - ((5.405e-4*T**2)/(T+204))
        EgAlGaAs = lambda x,T: 1.155*x + 0.37*x**2 - ((5.405e-4*T**2)/(T+204))
        EgAlAs   = lambda T: 2.239 - (6e-4*T**2/(408+T))
        #Functions to calculate Effective Mass as a function of composition
        meGaAs  = 0.067
        mlhGaAs = 0.083
        mhhGaAs = 0.55
        meAlAs  = 0.15
        mlhAlAs = 0.16
        mhhAlAs = 0.81
        meAlGaAs  = lambda x: (meAlAs*meGaAs)/(x*meGaAs + (1-x)*meAlAs)
        mlhAlGaAs = lambda x: (mlhAlAs*mlhGaAs)/(x*mlhGaAs + (1-x)*mlhAlAs)
        mhhAlGaAs = lambda x: (mhhAlAs*mhhGaAs)/(x*mhhGaAs + (1-x)*mhhAlAs)
        Qc = 0.6
        Qv = 0.4
        
        
        for layer in self.material:
            startindex  = round2int(position*1e-9/dx)
            position   += layer[0] # update position to end of the layer
            finishindex = round2int(position*1e-9/dx)
            matType     = layer[1]  
            x           = layer[2]
#             if matType in material_property:
#                 matprops = material_property[matType]
#                 Vc[startindex:finishindex]       =  0 #Joule
#                 eps[startindex:finishindex]      = matprops['epsilonStatic']*eps0
#                 cb_meff[startindex:finishindex]  = matprops['m_e']*m_e        
#             elif matType in alloy_property:
#                 alloyprops  = alloy_property[matType]
#                 mat1        = material_property[alloyprops['Material1']]
#                 mat2        = material_property[alloyprops['Material2']]
#                 Vc[startindex:finishindex]  = (alloyprops['Band_offset']*1.247*x)*q#((x*EgAlAs(T) + (1-x)*EgGaAs(T)))*q
#                 #(alloyprops['Band_offset']*1.247*x)*q #alloyprops['Band_offset']*(x*mat1['Eg'] + (1-x)* mat2['Eg'])*q
#                 eps[startindex:finishindex] = (x*mat1['epsilonStatic'] + (1-x)* mat2['epsilonStatic'] )*eps0
#                 cb_meff_alloy = x*mat1['m_e'] + (1-x)* mat2['m_e']
#                 cb_meff[startindex:finishindex] = cb_meff_alloy*m_e
            # doping
        
            if matType == 'GaAs':
                CB[startindex:finishindex] = (EgGaAs(T))*q
                VB[startindex:finishindex] = 0
                cb_meff[startindex:finishindex]  = meGaAs*m_e
                vblh_meff[startindex:finishindex] = mlhGaAs*m_e
                vbhh_meff[startindex:finishindex] = mhhGaAs*m_e
        
            elif matType == 'AlGaAs':
                CB[startindex:finishindex] = (EgGaAs(T)+EgAlGaAs(x,T)*Qc)*q
                VB[startindex:finishindex] =  -(EgAlGaAs(x,T)*Qv)*q
                cb_meff[startindex:finishindex]   = meAlGaAs(x)*m_e
                vblh_meff[startindex:finishindex] = mlhAlGaAs(x)*m_e
                vbhh_meff[startindex:finishindex] = mhhAlGaAs(x)*m_e
            elif matType == 'Vacuum':
                CB[startindex:finishindex] = (EgGaAs(T)+0.5)*q
                VB[startindex:finishindex] =  -(EgGaAs(T)+0.5)*q
                cb_meff[startindex:finishindex]   = m_e
                vblh_meff[startindex:finishindex] = m_e
                vbhh_meff[startindex:finishindex] = m_e
                
                
            if layer[4] == 'n':  
                chargedensity = layer[3]*1e6 #charge density in m**-3 (conversion from cm**-3)
            elif layer[4] == 'p': 
                chargedensity = -layer[3]*1e6 #charge density in m**-3 (conversion from cm**-3)
            else:
                chargedensity = 0.0
            
            dop[startindex:finishindex] = chargedensity
        
        self.CB         = CB
        self.VB         = VB
        self.cb_meff    = cb_meff
        self.vblh_meff  = vblh_meff
        self.vbhh_meff  = vbhh_meff
                


            
            
#--------------------------------- Schrodinger SOLVER 
def H(vpot,mass,dx,n):
    vb   = vpot 
    meff = mass
    H =np.zeros((n,n))
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

def Hholes(vpot,mass,dx,n):
    vb   = vpot 
    meff = mass
    H =np.zeros((n,n))
    m_minus  = (meff[0] + meff[1])/2.0
    m_plus   = (meff[0] + meff[1]) /2.0
    sn_minus =  pow(hbar/dx,2)/(2.0*m_minus)
    sn_plus  =  pow(hbar/dx,2)/(2.0*m_plus)
    bi       =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    H[0,0]  = bi - vb[0]
    H[0,1]  = sn_plus
    for ii in range (1,n-1):
        m_minus      = (meff[ii] + meff[ii-1])/2.0
        m_plus       = (meff[ii+1]    + meff[ii])/2.0
        sn_minus     = pow(hbar/dx,2)/(2.0*m_minus)
        sn_plus      = pow(hbar/dx,2)/(2.0*m_plus)
        bi           =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
        H[ii,ii-1]  = sn_minus
        H[ii,ii]    = bi - vb[ii] 
        H[ii,ii+1]  = sn_plus
    m_minus     = (meff[n-2] + meff[n-1])/2.0
    m_plus      = (meff[n-1] +  meff[n-2])/2.0
    sn_minus    =  pow(hbar/dx,2)/(2*m_minus)
    sn_plus     =  pow(hbar/dx,2)/(2*m_plus)
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
    pote     = model.CB + Fapp*q*xaxis
    poth     = model.VB + Fapp*q*xaxis
    me       = model.cb_meff
    Binding  = model.Binding
    mlh      = model.vblh_meff
    mhh      = model.vbhh_meff
    Emin     = model.Emin
    Emax     = model.Emax
    delta    = model.delta
    T        = model.T
    Ee    = np.zeros(subbands)
    Elh   = np.zeros(subbands)
    Ehh   = np.zeros(subbands)
    Psie  = np.zeros((subbands,n))
    Psihh = np.zeros((subbands,n))
    Psilh = np.zeros((subbands,n))
    EHH   = np.zeros(subbands)
    ELH   = np.zeros(subbands)
    Hamiltonian_e  = H(pote,me,dx,n)
    Hamiltonian_lh = Hholes(poth,mlh,dx,n)
    Hamiltonian_hh = Hholes(poth,mhh,dx,n)
    meV = 1e3
    
    PCB = pote
    PVB = poth
    

    
    if sparse == False:
        Ene,WFe   = eigh(Hamiltonian_e)
        Enhh,WFhh = eigh(Hamiltonian_hh)
        Enlh,WFlh = eigh(Hamiltonian_lh)
        for i in range(subbands):
            Ee[i]    = Ene[i]/q  # eV
            Elh[i]   = Enlh[i]/q 
            Ehh[i]   = Enhh[i]/q
            potcb    = PCB[:]/q
            potvb    = PVB[:]/q
    
        # Export Wavefunctions
        if absolute == False:
            for i in range(subbands):
                Psie[i]  = WFe[:,i]
                Psihh[i] = WFhh[:,i]
                Psilh[i] = WFlh[:,i]
        else:
            for i in range(subbands):
                Psie[i]  = np.power(np.absolute(WFe[:,i]),2)
                Psihh[i] = np.power(np.absolute(WFhh[:,i]),2)
                Psilh[i] = np.power(np.absolute(WFlh[:,i]),2)
            
            PSIhhMax = np.zeros(int(ens),dtype = np.float32)
            PSIlhMax = np.zeros(int(ens),dtype = np.float32)
            for k in range(subbands): 
                PSIhhMax[k] = np.amax(Psihh[k,:])
                PSIlhMax[k] = np.amax(Psilh[k,:])  
                
                
            
    else:
        #To Electrons
        upper_e     = np.diag(Hamiltonian_e,k=1)
        diagonal_e  = np.diag(Hamiltonian_e,k=0)
        lower_e     = np.diag(Hamiltonian_e,k=-1)
        H_diags_e = [-lower_e,diagonal_e,-upper_e]
        Hn_e = scipy.sparse.diags(H_diags_e, [-1,0,1], format='csc')
        energy_e,wfe_e = eigs(Hn_e, k= model.subbands, sigma = 0.0)
        energy_e = energy_e.real
        wfe_e = wfe_e.real
        #To Light Holes
        upper_lh     = np.diag(Hamiltonian_lh,k=1)
        diagonal_lh  = np.diag(Hamiltonian_lh,k=0)
        lower_lh     = np.diag(Hamiltonian_lh,k=-1)
        H_diags_lh = [-lower_lh,diagonal_lh,-upper_lh]
        Hn_lh = scipy.sparse.diags(H_diags_lh, [-1,0,1], format='csc')
        energy_lh,wfe_lh = eigs(Hn_lh, k= model.subbands, sigma = 0.0)
        energy_lh = energy_lh.real
        wfe_lh = wfe_lh.real
        #To Heavy Holes
        upper_hh     = np.diag(Hamiltonian_hh,k=1)
        diagonal_hh  = np.diag(Hamiltonian_hh,k=0)
        lower_hh     = np.diag(Hamiltonian_hh,k=-1)
        H_diags_hh = [-lower_hh,diagonal_hh,-upper_hh]
        Hn_hh = scipy.sparse.diags(H_diags_hh, [-1,0,1], format='csc')
        energy_hh,wfe_hh = eigs(Hn_hh, k= model.subbands, sigma = 0.0)
        energy_hh = energy_hh.real
        wfe_hh = wfe_hh.real
        
        if absolute == False:
            for i in range(subbands):
                Psie[i]  = wfe_e[:,i]
                Psilh[i] = wfe_lh[:,i]
                Psihh[i] = wfe_hh[:,i]
        else:
            for i in range(subbands):
                Psie[i]  = np.power(np.absolute(wfe_e[:,i]),2)
                Psilh[i] = np.power(np.absolute(wfe_lh[:,i]),2)
                Psihh[i] = np.power(np.absolute(wfe_hh[:,i]),2)
                
            PSIhhMax = np.zeros(subbands,dtype = np.float32)
            PSIlhMax = np.zeros(subbands,dtype = np.float32)
            for k in range(subbands): 
                PSIhhMax[k] = np.amax(Psihh[k,:])
                PSIlhMax[k] = np.amax(Psilh[k,:])  
            
        for i in range(subbands):
            Ee[i] = energy_e[i]/q  # eV
            Elh[i] = energy_lh[i]/q
            Ehh[i] = energy_hh[i]/q
            potcb  = PCB/q
            potvb  = PVB/q

        
    for i in range(subbands):
        EHH[i] = Ee[i]+Ehh[i]-Binding
        ELH[i] = Ee[i]+Elh[i]-Binding
        display(Math(r'\text{Transition} \,E_{%d}-HH_{%d}: %.4f'%(i,i,EHH[i])))
        display(Math(r'\text{Transition} \,E_{%d}-LH_{%d}: %.4f'%(i,i,ELH[i]))) 
    
    #print('Transicion $E_{i}$')
         
    # The solutions are degenerate: same energy
#     return {"E": E,
#             "Psi": Psi,
#             "xaxis":xaxis,
#             }
    

#     def LineShape(E,EHH,PSImax,Gamma):
#         ####3
#         return PSImax * E**2*np.exp(-(EHH-E)/(kBe*T))*np.exp(-(EHH-E)**2/(2*Gamma**2)) 
    
    LS = lambda E,EHH,PSImax,G : PSImax*E**2*np.exp(-(-EHH+E)/(kBe*T))*np.exp(-(-EHH+E)**2/(2*G**2)) 
    sFLH    = 1.1e-3#43#+(0.0064)/(math.exp((0.036)/(0.000086173324*Temperature))-1)
    sFLL    = 0.35e-3#+(0.0064)/(math.exp((0.036)/(0.000086173324*Temperature))-1)
    evec    = np.arange(Emin,Emax,delta)
    rangeFL = len(evec)
    FL = np.zeros(rangeFL,dtype=np.float64)
    for i in range(rangeFL):
        for j in range(subbands):
            auxE     = evec[i]
            auxPsihh = PSIhhMax[j]
            auxPsilh = PSIlhMax[j]
            auxEhh   = EHH[j]
            auxElh   = ELH[j]
            ihh = LS(auxE,auxEhh,auxPsihh,sFLH)
            ilh = LS(auxE,auxElh,auxPsilh,sFLH)
        FL[i] = ihh+ilh
    
#     aux = max(FL)  
#     for i in range(rangeFL):
#         FL[i] = FL[i]/max(FL)
        
        
        
        
        
    class Results(): pass
    results          = Results()
    results.xaxis    = xaxis
    results.Psie     = Psie
    results.Ee       = Ee
    results.Psilh    = Psilh
    results.Elh      = Elh
    results.Psihh    = Psihh
    results.Ehh      = Ehh
    results.CB       = potcb
    results.VB       = potvb
    results.dx       = dx
    results.subbands = subbands
    results.TEHH     = EHH
    results.TELH     = ELH
    results.evec     = evec
    results.PL       = FL
    return results




# def SolverSch(model,sparse = False,particle):
#     dx       = model.dx
#     n        = model.n_max
#     subbands = model.subbands 
#     xaxis    = np.arange(0,n)*dx 
#     Fapp     = model.Fapp
#     pote     = model.CB + Fapp*q*xaxis
#     poth     = model.VB + Fapp*q*xaxis
#     me       = model.cb_meff
#     mlh      = model.vblh_meff
#     mhh      = model.vbhh_meff
#     Ee    = np.zeros(subbands)
#     Elh   = np.zeros(subbands)
#     Ehh   = np.zeros(subbands)
#     Psie  = np.zeros((subbands,n))
#     Psihh = np.zeros((subbands,n))
#     Psilh = np.zeros((subbands,n))
    
#     if particle =='electron':
#         Hamiltonian  = H(pote,me,dx,n)
#     elif particle == 'heavyhole':
#         Hamiltonian = H(-poth,mlh,dx,n)
#     elif particle == 'lighthole':
#         Hamiltonian = H(-poth,mhh,dx,n)
    
#     if sparse == False:
#         En,WFe   = eigh(Hamiltonian)
#         for i in range(subbands):
#             Ee[i]    = Ene[i]  # JOULES
#             Psie[i]  = WFe[:,i]
    
#     else:
#         #To Electrons
#         upper      = np.diag(Hamiltonian,k=1)
#         diagonal   = np.diag(Hamiltonian,k=0)
#         lower      = np.diag(Hamiltonian,k=-1)
#         H_diags    = [-lower,diagonal,-upper]
#         Hn         = scipy.sparse.diags(H_diags, [-1,0,1], format='csc')
#         energy,wfe = eigs(Hn, k= model.subbands, sigma = 0.0)
#         energy     = energy.real
#         wfe = wfe_e.real
        
#         for i in range(subbands):
#             Ee[i] = energy[i]  # JOULES
#             Psi[i] = wfe[:,i]
        



#     # The solutions are degenerate: same energy
# #     return {"E": E,
# #             "Psi": Psi,
# #             "xaxis":xaxis,
# #             }
            
#     class Results(): pass
#     results = Results()
#     results.xaxis = xaxis
#     results.Psie = Psie
#     results.Ee   = Ee
#     results.CB  = pote
#     results.VB  = poth
#     results.dx = dx
#     results.subbands = subbands
#     return results
    