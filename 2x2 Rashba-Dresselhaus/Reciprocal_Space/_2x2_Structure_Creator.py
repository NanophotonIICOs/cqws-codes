# Import
import numpy as np

def round2int(x):
    return int(x+0.5)

# Constants and Parameters
q    = 1.602176e-19 #C
m_e  = 9.1093826E-31 #kg
hbar = 1.054571817e-34 #J s


# Electron Effective Mass
meGaAs = 0.0665
meff   = meGaAs*m_e  

# Gap
EgGaAs   = lambda T: 1.519 - (((0.5405e-3)*(T**2))/(204+T))
EgAlAs   = lambda T: 3.099 - (((0.885e-3)*(T**2))/(530+T))
EgAlGaAs = lambda x,T: 1.519 + 1.155*x + 0.37*x**2  - (((0.5405e-3)*(T**2))/(204+T))




def cr_InterfacesIndex(dx,Htr):
    finalI=np.alen(Htr)
    InterfacesIndex         = np.zeros(int((2)*(finalI-1)), dtype = int)
    position=0
    
    for i in range(finalI-1):
        position   += Htr[i][0]
        finishindex = round2int(position/dx)
        InterfacesIndex[2*(i)]=int(finishindex-1)
        InterfacesIndex[2*(i)+1]=int(finishindex)
    
    return InterfacesIndex




def cr_CB(T,dx,Htr):
    Ns=0
    aux=0
    for i in range(np.alen(Htr)):
        aux+=Htr[i][0]
    Ns=round2int(aux/dx)
    CB                = np.zeros(Ns)
    position=0
    for i in range(np.alen(Htr)):
        matType=Htr[i][1]
        x=Htr[i][2]
        startindex  = round2int(position/dx)
        position   += Htr[i][0]
        finishindex = round2int(position/dx)
        if matType == 'GaAs':
            CB[startindex:finishindex] =  0
            
        
        elif matType == 'AlAs':
            CB[startindex:finishindex] =  0.65*(EgAlAs(T)-EgGaAs(T))*q
            
        
        elif matType == 'AlGaAs':
            CB[startindex:finishindex] =  0.65*(EgAlGaAs(x,T)-EgGaAs(T))*q
            
    
    
    
    CB[:]=(CB[:])-(CB[:].max())
    
    
    return CB


def cr_Zaxis(T,dx,Htr):
    Ns=0
    aux=0
    for i in range(np.alen(Htr)):
        aux+=Htr[i][0]
    Ns=round2int(aux/dx)
    Zaxis=np.zeros(Ns)
    aux=0
    
    aux+=Htr[0][0]
    aux+=Htr[1][0]
    aux+=(Htr[2][0])/2
    rest=aux*1e-9
    for i in range(Ns):
        Zaxis[i]=i*dx*1e-9
    
    
    
    return Zaxis



def cr_Kzaxis(T,dx,dK,Htr):
    CB=cr_CB(T,dx,Htr)
    mid=CB.size-1
    Gs=np.zeros((2*mid+1))
    for i in range(-mid,mid+1):
        Gs[i+mid]=dK*i
    return Gs


def cr_V_k(T,dx,dK,Htr):
    
    Kzaxis=cr_Kzaxis(T,dx,dK,Htr)
    
    CB=cr_CB(T,dx,Htr)
    
    Zaxis=cr_Zaxis(T,dx,Htr)
    
    V_k=np.zeros((Kzaxis.size),dtype=complex)
    
    for i in range(Kzaxis.size):
        V_k[i]=(1/(2*np.pi))*np.trapz(((CB[:])*(np.cos((1)*(Kzaxis[i])*(Zaxis[:]))))-(1j)*((CB[:])*(np.sin((1)*(Kzaxis[i])*(Zaxis[:])))),Zaxis[:])
    
    
    return V_k


def cr_V_k_z(T,dx,dK,Htr):
    
    Kzaxis=cr_Kzaxis(T,dx,dK,Htr)
    
    V_k=cr_V_k(T,dx,dK,Htr)
    
    Zaxis=cr_Zaxis(T,dx,Htr)
    
    V_k_z=np.zeros((Zaxis.size))
    
    for i in range(Zaxis.size):
        V_k_z[i]=(1/(1))*np.real(np.trapz(((V_k[:])*(np.cos((1)*(Kzaxis[:])*(Zaxis[i]))))+(1j)*((V_k[:])*(np.sin((1)*(Kzaxis[:])*(Zaxis[i])))),Kzaxis[:]))
    #V_k_z[:]=(V_k_z[:])/(Kzaxis.size)
    
    return V_k_z



def cr_Kzaxis_2(T,dx,dK,Htr):
    CB=cr_CB(T,dx,Htr)
    mid=CB.size-1
    Gs=np.zeros((2*mid+1))
    for i in range(Gs.size):
        Gs[i]=-dK*i
    return Gs



def cr_V_k_2(T,dx,dK,Htr):
    
    Kzaxis=cr_Kzaxis_2(T,dx,dK,Htr)
    
    CB=cr_CB(T,dx,Htr)
    
    Zaxis=cr_Zaxis(T,dx,Htr)
    
    V_k=np.zeros((Kzaxis.size),dtype=complex)
    
    for i in range(Kzaxis.size):
        V_k[i]=(1/(2*np.pi))*np.trapz(((CB[:])*(np.cos((1)*(Kzaxis[i])*(Zaxis[:]))))-(1j)*((CB[:])*(np.sin((1)*(Kzaxis[i])*(Zaxis[:])))),Zaxis[:])
    
    
    return V_k









def cr_BorW(dx,Htr):
    Ns=0
    aux=0
    for i in range(np.alen(Htr)):
        aux+=Htr[i][0]
    Ns=round2int(aux/dx)
    BorW         = np.empty(Ns,dtype=str)
    position=0
    for i in range(np.alen(Htr)):
        matType=Htr[i][3]
        startindex  = round2int(position/dx)
        position   += Htr[i][0] # update position to end of the layer
        finishindex = round2int(position/dx)
        if matType == 'B':
            BorW[startindex:finishindex] = 'B'
        
        elif matType == 'W':
            BorW[startindex:finishindex] = 'W'
    
    return BorW














