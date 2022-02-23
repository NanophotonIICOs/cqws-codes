import numpy as np
from numba import jit, prange
import _2x2_Structure_Creator as str_cr


def round2int(x):
    return int(x+0.5)

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------



# Constants and Parameters
q    = 1.602176e-19 #C
m_e  = 9.1093826E-31 #kg
hbar = 1.054571817e-34 #J s

# Electron Effective Mass
meGaAs = 0.07
meff   = meGaAs*m_e  


def kll2(kx,ky):
    return (kx**2+ky**2)


def cr_HBIA(kx,ky,beta):
    
        
    HBIA=np.zeros((2,2),dtype=complex)
    HBIA[:] = [
            [0         ,kx+(1j)*ky],
            [kx-(1j)*ky,0]
           ]
    HBIA[:]=(beta)*(-1)*HBIA[:]
    
    return HBIA[:]



def cr_HSIA(kx,ky,alpha):
    
        
    HSIA=np.zeros((2,2),dtype=complex)
    HSIA[:] = [
            [0         ,ky+(1j)*kx],
            [ky-(1j)*kx,0]
           ]
    HSIA[:]=(alpha)*HSIA[:]
    
    return HSIA[:]






@jit(parallel=True)
def cr_Hamiltonian_Aux(N,kz,Vk,dK,k2,HBIA,HSIA,Hamiltonian):
    for jj in prange(N):
        for ii in prange(jj+1):
            if (ii==jj):
                for i2 in range(2):
                    #---------------H01------------------------------
                    Hamiltonian[2*(ii)+i2,2*(ii)+i2]+=k2
                    #---------------H02-----------------------------
                    Hamiltonian[2*(ii)+i2,2*(ii)+i2]+=kz[ii]
                    for j2 in range(2):
                        #---------------HBIA-----------------------------
                        Hamiltonian[2*(ii)+i2,2*(ii)+j2]+=HBIA[i2,j2]
                        #---------------HSIA------------------------------
                        Hamiltonian[2*(ii)+i2,2*(ii)+j2]+=HSIA[i2,j2]
            for i2 in range(2):
                #---------------H00------------------------------
                Hamiltonian[2*(ii)+i2,2*(jj)+i2]+=(Vk[jj-ii])*dK
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    return Hamiltonian[:]






def cr_Hamiltonian(T,dz,dK,Htr,alpha,beta,kx,ky):
    
    kz=str_cr.cr_Kzaxis(T,dz,dK,Htr)
    
    N=kz.size
    NH=(kz.size)*2
    
    kz[:]=((hbar**2)/(2*meff))*((kz[:])**2)
    
    Vk=str_cr.cr_V_k_2(T,dz,dK,Htr)
    
    k2=((hbar**2)/(2*meff))*(kll2(kx,ky))
    
    Hamiltonian=np.zeros((NH,NH),dtype=complex)
    
    HBIA=np.zeros((2,2),dtype=complex)
    HSIA=np.zeros((2,2),dtype=complex)
    
    HBIA=cr_HBIA(kx,ky,beta)
    HSIA=cr_HSIA(kx,ky,alpha)
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    return cr_Hamiltonian_Aux(N,kz[:],Vk[:],dK,k2,HBIA[:],HSIA[:],Hamiltonian[:])
