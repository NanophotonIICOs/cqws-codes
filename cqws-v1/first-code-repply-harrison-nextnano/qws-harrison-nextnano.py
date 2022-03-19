# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:58:53 2018

@author: Oscar Ruiz Cigarrillo
""" 
#============== NUMBA AND NUMPY ==============
import sys
import numpy as np
import numba
from numpy import linalg as LA
#============= MATPLOTLIB=====================
import matplotlib
import scipy.constants 
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pylab as plt
import pandas as pd
#=============== Warnings =====================
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
#=============== Time and Date =================
import timeit
start_time = timeit.default_timer()
from datetime import date
from datetime import datetime
today = date.today()

#=============== BARS TQDM =====================
from tqdm import tqdm
from tabulate import tabulate
# Use LaTeX throughout the figure for consistency
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
#rc('text', usetex=True)
# =========================== PARAMETERS ==================================
print('PROGRAM STARTS ON %s' %today+'  AT %s' %datetime.time(datetime.now()))
sample = 'QWS-HARRISON-NEXTNANO'
print('SAMPLE: '+sample)
nm  = 1.0e-9 
b1  = 20*nm
qw1 = 6*nm
b2  = 4*nm
qw2 = 6*nm
b3  = 20*nm
L   = b1+qw1+b2+qw2+b3
N   = 2520 # No. of points 
xx  = 0.0
dx  = L/N
x   = np.linspace(0, L, N)

# CONSTANTS
Vcb  =  np.zeros(N,dtype = np.float32)
Vvb  =  np.zeros(N,dtype = np.float32)
e0   = scipy.constants.e
hbar = scipy.constants.hbar
m0   = scipy.constants.m_e

meV  = 1.0e-3
eV2J = e0
J2eV = 1./eV2J
Qc = 0.67
Qv = 0.33
amp  = 45
Temperature = 300
xcomp    = 0.2


# COLORS#
COLORS = ['r','b','sienna','darkcyan','olive','navy','teal','gold','royalblue','m','limegreen']
cols = 120
# Materials
b1m   = 'Al(%.2f)'%xcomp + 'Ga(%.2f)'%(1-xcomp)+'As'
qw1m  = 'GaAs'
b2m   = 'Al(%.2f)'%xcomp + 'Ga(%.2f)'%(1-xcomp)+'As'
qw2m  = 'GaAs'
b3m   = 'Al(%.2f)'%xcomp + 'Ga(%.2f)'%(1-xcomp)+'As'
db1s  = '%3.3f nm'%(b1/nm)
dqw1s = '%3.3f nm'%(qw1/nm)
db2s  = '%3.3f nm'%(b2/nm)
dqw2s = '%3.3f nm'%(qw2/nm)
db3s  = '%3.3f nm'%(b3/nm)



sp = ' '
print('SAMPLE STRUCTURE:\n')
print(tabulate([[sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp],
                [db1s,dqw1s,db2s,dqw2s,db2s],
                [sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp]],
                [b1m ,qw1m , b2m,qw2m, b3m ], tablefmt="pipe",stralign='center',))
print('\n')
    
# ========================== BAND STRUCTURE ===============================

@numba.jit
def STRUCTURE(pt):
    if   ( 0 <  pt <= b1):
         return 1
    elif (  b1 <= pt <= b1+qw1):
         return 0
    elif (b1+qw1 <= pt <= b1+qw1+b2):
         return 1
    elif (b1+qw1+b2 <= pt <= b1+qw1+b2+qw2):
         return 0
    elif (b1+qw1+b2+qw2 <= pt<= b1+qw1+b2+qw2+b3):
         return 1
    else:
        return 1
    

def VCB(pt):
    xcomp    = 0.2
    DeltaEg  =  (1.2470*xcomp)*e0
    if   (STRUCTURE(pt)==1):
            return  DeltaEg*Qc
    else:
            return  0
   

    
# ========================== EFFECTIVE MASS  ===============================
 
    
# Nakwaski, W. (1995). Effective masses of electrons and heavy holes in GaAs, 
#InAs, A1As and their ternary compounds. Physica B: Condensed Matter, 210(1), 1-25.
#Photoluminescence of AlxGa1−xAs alloys,Lorenzo Pavesi and Mario Guzzi

    
# Electron Effective Mass, 
def effmee(pt): 
     mAlGaAs  = (0.067+0.083*xcomp)*m0
     mGaAs    =  0.067*m0
     if    (STRUCTURE(pt) == 1):
            return mAlGaAs 
     elif (STRUCTURE(pt) == 0):
           return mGaAs
     else: 
           return mAlGaAs 
            

        

                              
        
# ====================== QUANTUM WELLS POTENTIAL =============================
xx = 0.0
for i in tqdm(range(N), desc = 'Conduction Band vector        ',total=N,ascii = False,ncols = 100):
    xx = xx + dx
    Vcb[i] =  J2eV*VCB(xx)
    #sleep(1e-15)    
#%================== HAMILTONIAN FINITE DIFFERENCE METHOD ======================

#-----------------------------------------------------------------------------#
#                        ELECTRON EIGENVALUES                                 #
#-----------------------------------------------------------------------------#
def HAMILTONIAN_e():
    xx       = 0.0
    hbar     = scipy.constants.hbar
    He       = np.zeros((N,N),dtype = np.float32)
    m_minus  = (effmee(xx) + effmee(xx-dx))/2.0
    m_plus   = (effmee(xx+dx)    + effmee(xx))/2.0
    sn_minus =  -pow(hbar/dx,2)/(2.0*m_minus)
    sn_plus  =  -pow(hbar/dx,2)/(2.0*m_plus)
    bi       =  0.5*pow(hbar/dx,2)*(m_plus+m_minus)/(m_plus*m_minus)
    He[0,0]  = bi + VCB(xx)
    He[0,1]  = sn_plus
    for ii in tqdm(range(1,N-2), desc = 'Electron  Hamiltoninan Matrix ', total=N, ascii = False,ncols = 100):
        xx           = xx+dx
        m_minus      = (effmee(xx) + effmee(xx-dx))/2.0
        m_plus       = (effmee(xx+dx)    + effmee(xx))/2.0
        sn_minus     =  -pow(hbar/dx,2)/(2.0*m_minus)
        sn_plus      =  -pow(hbar/dx,2)/(2.0*m_plus)
        bi           =  0.5*pow(hbar/dx,2)*(m_plus+m_minus)/(m_plus*m_minus)
        He[ii,ii-1]  = sn_minus
        He[ii,ii]    = bi + VCB(xx)
        He[ii,ii+1]  = sn_plus
    xx          = xx+dx
    m_minus     = (effmee(xx) + effmee(xx-dx))/2.0
    m_plus      = (effmee(xx+dx)    + effmee(xx))/2.0
    sn_minus    =  -pow(hbar/dx,2)/(2.0*m_minus)
    sn_plus     =  -pow(hbar/dx,2)/(2.0*m_plus)
    bi          =  0.5*pow(hbar/dx,2)*(m_plus+m_minus)/(m_plus*m_minus)
    He[N-1,N-1] = bi + VCB(xx)
    He[N-1,N-2] = sn_minus
    [eigvals,eigvec] = LA.eigh(He)
    return [eigvals,eigvec]
 
[eigvals_e,eigvec_e] = HAMILTONIAN_e()
Ee = np.zeros(N,dtype = np.float32)

for m in tqdm(range(N), desc = 'Electron  eigenenergies vector',total=N, ascii = False,ncols = 100):
    Ee[m] = J2eV*eigvals_e[m]
    
end = timeit.default_timer()
if (end - start_time) < 60:
    print("CPU TIME(s) :\t %.4s"  %(end - start_time))
else:
    print("CPU TIME(min) : \t %.4s" %((end - start_time)/60))


ens = input('No. Of Transition Energies (>0)?===> ')

if int(ens) == 0:
    print('Error you select 0')
    sys.exit(0)
    
eprint = np.zeros((int(ens),2),dtype = object)
for ii in range(1,int(ens)+1):
    eprint[ii-1,:] = ['E'+str(ii) , '%.5f'%Ee[ii]] 
    
    
    
print('\n')
print(tabulate(eprint,['# Energy','eV'], tablefmt="pipe",stralign='center',))
print('\n')

    
trans_file = open('data/ENERGIES.txt','w')
trans_file.write('PROGRAM STARTS ON %s' %today+'  AT %s' %datetime.time(datetime.now()))
trans_file.write('\nTHIS PROGRAM CALCULATE OPTICAL TRANSITIONS OF SAMPLE: '+sample)
trans_file.write('\nEMPLOYED PARAMETERS IN THIS CALCULATIONS: ')
trans_file.write('\n')
trans_file.write(tabulate([['TEMPERATURE(K):',Temperature],
                           ['x COMPOSITION',xcomp],
                           ['BAND OFFSET',str(str(Qc)+':'+str(Qv))]], tablefmt="simple",stralign='right'))
trans_file.write('\n')
#trans_file.write('\n TEMPERATURE\t: %1d (K)' %Temperature)
#trans_file.write('\n x COMPOSITION\t: %.2f' %xcomp)
#trans_file.write('\n BAND OFFSET\t: %.2f:%.2f' %(Qc,Qv))
trans_file.write('\n SAMPLE STRUCTURE: \n')
trans_file.write(tabulate([[sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp],
                [db1s,dqw1s,db2s,dqw2s,db2s],
                [sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp]],
                [b1m ,qw1m , b2m,qw2m, b3m ], tablefmt="pipe",stralign='center'))
trans_file.write('\n')
trans_file.write('\n\n ELECTRON ENERGIES:')
# ENRGIES
trans_file.write('\n')
trans_file.write(tabulate(eprint,['# STATE','eV'], tablefmt="pipe",stralign='center',))
trans_file.write('\n')
end = timeit.default_timer()
if (end - start_time) < 60:
    print(" CPU TIME(s)\t : %.2f"  %(end - start_time))
    trans_file.write('\n\nCPU TIME (s) : \t %.2f'%(end - start_time))
else:
    print(" CPU TIME(min)\t : %.4s" %((end - start_time)/60))
    trans_file.write("\n\nCPU TIME(min) :  %.2f" %((end - start_time)/60))
trans_file.close()


#============================ EXPORT DATA ==========================

xmindata =  int(0)
xmaxdata =  int((b1+qw1+b2+qw2 +b3)/nm)
ran      = xmaxdata-xmindata
PSI      = np.zeros((int(ens),N),dtype = np.float32)






export = input('\nDo you want export data? (y/n) ===> ')
if (export == 'y'):
    absolute =  input('Do you prefer absolute wave functions? (y/n) ===> ')
    if (absolute == 'y'):
        for k in tqdm(range(1,int(ens)+1), desc = 'Generating Wave Functions Vectors      ', ascii = False,ncols =100):
            PSI[k-1,:]  = amp*eigvec_e[:,k]**2+Ee[k]
    else:
        for k in tqdm(range(1,int(ens)+1),desc = 'Generating Wave Functions Vectors      ', ascii = False,ncols = 100):
            PSI[k-1,:]  = amp*eigvec_e[:,k]+Ee[k]
    # POTENTIAL PROFILES
    with open("data/Potential_Profile" + ".dat", "w") as out_file:
           for i in tqdm(range(N),desc = 'Exporting potential profile data       ', ascii = False,ncols = 100):
               out_string = ""
               out_string += str(x[i]/nm)
               out_string += ","  + str(Vcb[i])
               out_string += ","  + str(Vvb[i])   
               out_string += "\n"
               out_file.write(out_string)
    # Wave Function electron 
    with open("data/PSIelectron_data" + ".dat", "w") as out_file:
        for i in tqdm(range(N),desc = 'Exporting Electron  Wave Functions data', ascii = False,ncols = 100):
            out_string = ""
            out_string += str(x[i]/nm)
            for j in range(int(ens)):
                out_string += ","  + str(f'{PSI[j,i]:.4f}')
            out_string += "\n"  
            out_file.write(out_string)
    end = timeit.default_timer()
    if (end - start_time) < 60:
        print("CPU TIME(s) :\t %.2f"  %(end - start_time))
    else:
        print("CPU TIME(min) : \t %.2f" %((end - start_time)/60))
else :
    print('Ok')
    end = timeit.default_timer() 
    if (end - start_time) < 60:
        print("CPU TIME(s) :\t %.2f"  %(end - start_time))
    else:
        print("CPU TIME(min) : \t %.2f" %((end - start_time)/60))
    




#
#plt.rc('font', family='serif',size=10 )
#fig,ax =  plt.subplots(nrows=1, ncols =1 , figsize = (5,5))
#fig.tight_layout()
##fig.subplots_adjust(bottom=0.1,right = 0.1)
##plt.margins(0.5)
#ax.plot(x/nm, Vcb,'-k')
#ax.plot(x/nm, psi1,'-b',lw = 1)
#ax.plot(x/nm, psi2,'-r',lw = 1)
#ax.set_xlabel('$z$ [\AA]',size = 15)
#ax.set_ylabel('$\psi$',size = 15)
#
#plt.subplots_adjust(bottom=0.1, right=0.9, top=0.95,left = 0.2)
#plt.show()

#plt.plot(x,PSI_bonding)
#plt.plot(x,PSI_anti_bonding)
# ============================== PLOT =========================================
#
