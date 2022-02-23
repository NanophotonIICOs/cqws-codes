#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
"""
This library is used to assemble the 2Nx2N Hamiltonian which take into account the Schrödinger equation and
the spin-orbit (SO) interaction of Dresselhaus and Rashba terms in a Coupled Quantum Well.

It is used to find the E-k dispersion for a specific direction.

At first the SO interactions emerge in Asymmetric Quantum Wells.
"""
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
import numpy as np
import _2x2_Structure_Creator as str_cr


def round2int(x):
    return int(x+0.5)

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------



# Constants and Parameters
q    = 1.602176e-19 #C
m_e  = 9.1093826E-31 #kg
hbar = 1.054571817e-34 #J s
M=(hbar**2)/(2*m_e)

def kll2(kx,ky):
    return (kx**2+ky**2)



def cr_H00(Ec):
    H00=np.zeros((2,2),dtype=complex)
    H00[:] = [
            [Ec,0],
            [0 ,Ec]
           ]
    
    
    return H00[:]



def cr_H01(meff,kx,ky):
    k2=kll2(kx,ky)
        
    H01=np.zeros((2,2),dtype=complex)
    H01[:] = [
            [((hbar**2)/(2*meff))*(k2),0],
            [0                        ,((hbar**2)/(2*meff))*(k2)]
           ]
    
    
    return H01[:]



def cr_H02(meff):
        
    H02=np.zeros((2,2),dtype=complex)
    H02[:] = [
            [((hbar**2)/(2*meff)),0],
            [0                   ,((hbar**2)/(2*meff))]
           ]
    
    
    return H02[:]



def cr_HBIA(kx,ky):
    
        
    HBIA=np.zeros((2,2),dtype=complex)
    HBIA[:] = [
            [0         ,kx+(1j)*ky],
            [kx-(1j)*ky,0]
           ]
    HBIA[:]=(-1)*HBIA[:]
    
    return HBIA[:]




def cr_HSIA(kx,ky):
    
        
    HSIA=np.zeros((2,2),dtype=complex)
    HSIA[:] = [
            [0         ,ky+(1j)*kx],
            [ky-(1j)*kx,0]
           ]
    
    return HSIA[:]



#-----------------------------------------------------------------------------------------------------------
"""
In here starts the Hamiltonian Creator.
"""
#-----------------------------------------------------------------------------------------------------------

def cr_Hamiltonian(T,dz,Htr,alpha,beta,kx,ky):
    
    z=str_cr.cr_Zaxis(T,dz,Htr)
    N=z.size
    NH=(z.size)*2
    
    Parameters=str_cr.cr_Parameters(T,dz,Htr)
    
    
    Hamiltonian=np.zeros((NH,NH),dtype=complex)
    
    
    for ii in range(1,N-1):#Principal terms of the tridiagonal
        
        
        meffAnt=Parameters[2,ii-1]
        
        
        Ec=Parameters[1,ii]
        meff=Parameters[2,ii]
        
        meffDesp=Parameters[2,ii+1]
        
        
        H02Ant=np.zeros((2,2),dtype=complex)
        
        H00=np.zeros((2,2),dtype=complex)
        H01=np.zeros((2,2),dtype=complex)
        H02=np.zeros((2,2),dtype=complex)
        HBIA=np.zeros((2,2),dtype=complex)
        HSIA=np.zeros((2,2),dtype=complex)
        
        H02Desp=np.zeros((2,2),dtype=complex)
        
        
        
        
        H02Ant[:]=cr_H02(meffAnt)
        
        H00[:]=cr_H00(Ec)
        H01[:]=cr_H01(meff,kx,ky)
        H02[:]=cr_H02(meff)
        HBIA[:]=cr_HBIA(kx,ky)
        HSIA[:]=cr_HSIA(kx,ky)
        
        H02Desp[:]=cr_H02(meffDesp)
        
        
        
        
        
        for i2 in range(2):
            for j2 in range(2):
                #---------------H00------------------------------
                Hamiltonian[2*(ii)+i2,2*(ii)+j2]+=H00[i2,j2]
                #---------------H01------------------------------
                Hamiltonian[2*(ii)+i2,2*(ii)+j2]+=H01[i2,j2]
                #---------------H02-----------------------------
                Hamiltonian[2*(ii)+i2,2*(ii-1)+j2]+=(-1)*((1)/(2*(dz*1e-9)**2))*(H02Ant[i2,j2]+H02[i2,j2])
                Hamiltonian[2*(ii)+i2,2*(ii)+j2]+=(-1)*((1)/(2*(dz*1e-9)**2))*(-1)*(H02Ant[i2,j2]+2*H02[i2,j2]+H02Desp[i2,j2])
                Hamiltonian[2*(ii)+i2,2*(ii+1)+j2]+=(-1)*((1)/(2*(dz*1e-9)**2))*(H02[i2,j2]+H02Desp[i2,j2])
                #---------------HBIA-----------------------------
                Hamiltonian[2*(ii)+i2,2*(ii)+j2]+=(beta)*HBIA[i2,j2]
                #---------------HSIA------------------------------
                Hamiltonian[2*(ii)+i2,2*(ii)+j2]+=(alpha)*(HSIA[i2,j2])
                
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    
        
    Ec=Parameters[1,0]
    meff=Parameters[2,0]
    
    meffDesp=Parameters[2,1]
    
    H00=np.zeros((2,2),dtype=complex)
    H01=np.zeros((2,2),dtype=complex)
    H02=np.zeros((2,2),dtype=complex)
    HBIA=np.zeros((2,2),dtype=complex)
    HSIA=np.zeros((2,2),dtype=complex)
    
    H02Desp=np.zeros((2,2),dtype=complex)
    
    
    H00[:]=cr_H00(Ec)
    H01[:]=cr_H01(meff,kx,ky)
    H02[:]=cr_H02(meff)
    HBIA[:]=cr_HBIA(kx,ky)
    HSIA[:]=cr_HSIA(kx,ky)
    
    H02Desp[:]=cr_H02(meffDesp)
    
    
    #------------------------------------------------------------------
    H0F=np.zeros((2,2),dtype=complex)
    H0F[:] = [
            [((hbar**2)/(2*m_e)),0],
            [0                   ,((hbar**2)/(2*m_e))]
           ]
    #------------------------------------------------------------------
    
    
    
    
    for i2 in range(2):#The n=0 boundary
        for j2 in range(2):
            #---------------H00------------------------------
            Hamiltonian[i2,j2]+=H00[i2,j2]
            #---------------H01------------------------------
            Hamiltonian[i2,j2]+=H01[i2,j2]
            #---------------H02-----------------------------
            Hamiltonian[i2,j2]+=(-1)*((1)/(2*(dz*1e-9)**2))*(-1)*(H0F[i2,j2]+2*H02[i2,j2]+H02Desp[i2,j2])#H0F[i2,j2]+
            Hamiltonian[i2,2*(1)+j2]+=(-1)*((1)/(2*(dz*1e-9)**2))*(H02[i2,j2]+H02Desp[i2,j2])
            #---------------HBIA-----------------------------
            Hamiltonian[i2,j2]+=(beta)*HBIA[i2,j2]
            #---------------HSIA------------------------------
            Hamiltonian[i2,j2]+=(alpha)*(HSIA[i2,j2])
    
    #-----------------------------------------------------------------------------------------------
    
    meffAnt=Parameters[2,(N-1)-1]
    
    Ec=Parameters[1,(N-1)]
    meff=Parameters[2,(N-1)]
    
    H02Ant=np.zeros((2,2),dtype=complex)
    
    H00=np.zeros((2,2),dtype=complex)
    H01=np.zeros((2,2),dtype=complex)
    H02=np.zeros((2,2),dtype=complex)
    HBIA=np.zeros((2,2),dtype=complex)
    HSIA=np.zeros((2,2),dtype=complex)
    
    
    
    H02Ant[:]=cr_H02(meffAnt)
    
    H00[:]=cr_H00(Ec)
    H01[:]=cr_H01(meff,kx,ky)
    H02[:]=cr_H02(meff)
    HBIA[:]=cr_HBIA(kx,ky)
    HSIA[:]=cr_HSIA(kx,ky)
    
    
    for i2 in range(2):#The n=N-1 boundary
        for j2 in range(2):
            #---------------H00------------------------------
            Hamiltonian[2*((N-1))+i2,2*((N-1))+j2]+=H00[i2,j2]
            #---------------H01------------------------------
            Hamiltonian[2*((N-1))+i2,2*((N-1))+j2]+=H01[i2,j2]
            #---------------H02-----------------------------
            Hamiltonian[2*((N-1))+i2,2*((N-1)-1)+j2]+=(-1)*((1)/(2*(dz*1e-9)**2))*(H02Ant[i2,j2]+H02[i2,j2])
            Hamiltonian[2*((N-1))+i2,2*((N-1))+j2]+=(-1)*((1)/(2*(dz*1e-9)**2))*(-1)*(H02Ant[i2,j2]+2*H02[i2,j2]+H0F[i2,j2])#+H0F[i2,j2]
            #---------------HBIA-----------------------------
            Hamiltonian[2*((N-1))+i2,2*((N-1))+j2]+=(beta)*HBIA[i2,j2]
            #---------------HSIA-----------------------------
            Hamiltonian[2*((N-1))+i2,2*((N-1))+j2]+=(alpha)*(HSIA[i2,j2])
            
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    auxReal=0
    auxImag=0
    for ii in range(1,N-1):#Here average interfaces to ensure continuity and hermicity
        for i2 in range(2):
            for j2 in range(2):
                auxReal=0
                auxImag=0
                auxReal+=np.real(Hamiltonian[2*(ii-1)+i2,2*(ii)+j2])
                auxReal+=np.real(Hamiltonian[2*(ii)+j2,2*(ii-1)+i2])
                auxReal=auxReal/2
                auxImag+=np.abs(np.imag(Hamiltonian[2*(ii-1)+i2,2*(ii)+j2]))
                auxImag+=np.abs(np.imag(Hamiltonian[2*(ii)+j2,2*(ii-1)+i2]))
                auxImag=auxImag/2
                if (np.imag(Hamiltonian[2*(ii-1)+i2,2*(ii)+j2])>=0):
                    Hamiltonian[2*(ii-1)+i2,2*(ii)+j2]=auxReal+(1j)*auxImag
                    Hamiltonian[2*(ii)+j2,2*(ii-1)+i2]=auxReal-(1j)*auxImag
                elif (np.imag(Hamiltonian[2*(ii-1)+i2,2*(ii)+j2])<0):
                    Hamiltonian[2*(ii-1)+i2,2*(ii)+j2]=auxReal-(1j)*auxImag
                    Hamiltonian[2*(ii)+j2,2*(ii-1)+i2]=auxReal+(1j)*auxImag
    
    return Hamiltonian[:]
