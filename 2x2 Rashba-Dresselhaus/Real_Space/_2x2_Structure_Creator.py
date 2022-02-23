# Import
import numpy as np
           
def round2int(x):
    return int(x+0.5)

# Constants and Parameters
q    = 1.602176e-19 #C
m_e  = 9.1093826E-31 #kg
hbar = 1.054571817e-34 #J s




# Electron Effective Mass
meGaAs    = 0.0665
meAlAs    = 0.15
meAlGaAs  = lambda x:  (meAlAs*meGaAs)/(x*meGaAs + (1-x)*meAlAs)



# Gap
EgGaAs   = lambda T: 1.519 - (((0.5405e-3)*(T**2))/(204+T))
EgAlAs   = lambda T: 3.099 - (((0.885e-3)*(T**2))/(530+T))
EgAlGaAs = lambda x,T: 1.519 + 1.155*x + 0.37*x**2  - (((0.5405e-3)*(T**2))/(204+T))





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
    
    
    return CB




def cr_CBmeff(T,dx,Htr):
    Ns=0
    aux=0
    for i in range(np.alen(Htr)):
        aux+=Htr[i][0]
    Ns=round2int(aux/dx)
    
    CB_meff           = np.zeros(Ns)
    position=0
    for i in range(np.alen(Htr)):
        matType=Htr[i][1]
        x=Htr[i][2]
        startindex  = round2int(position/dx)
        position   += Htr[i][0]
        finishindex = round2int(position/dx)
        if matType == 'GaAs':
            CB_meff[startindex:finishindex]  = meGaAs*m_e
        
        elif matType == 'AlAs':
            CB_meff[startindex:finishindex]  =  meAlAs*m_e
        
        elif matType == 'AlGaAs':
            CB_meff[startindex:finishindex]   = meAlGaAs(x)*m_e
    
    return CB_meff


def cr_Zaxis(T,dx,Htr):
    Ns=0
    aux=0
    for i in range(np.alen(Htr)):
        aux+=Htr[i][0]
    Ns=round2int(aux/dx)
    Zaxis=np.zeros(Ns)
    for i in range(Ns):
        Zaxis[i]=i*dx*1e-9
    
    return Zaxis





def cr_Parameters(T,dx,Htr):
    Ns=0
    aux=0
    for i in range(np.alen(Htr)):
        aux+=Htr[i][0]
    Ns=round2int(aux/dx)
    Parameters         = np.zeros((3,Ns))
    
    Parameters[0,:]=cr_Zaxis(T,dx,Htr)
    Parameters[1,:]=cr_CB(T,dx,Htr)
    Parameters[2,:]=cr_CBmeff(T,dx,Htr)
    
    
    return Parameters



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














