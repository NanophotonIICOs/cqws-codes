# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:28:10 2018

@author: RUCO
"""

#============== NUMBA AND NUMPY ==============
import sys
import os
import numpy as np
import numba
from numpy import linalg as LA
#============= MATPLOTLIB=====================
import matplotlib
import scipy.constants
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pylab as plt
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


#======================== READ DATA =================================
import tkinter as tk
from tkinter import filedialog
#from tkinter import messagebox

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
#if file_path == ' ':
#    messagebox.showinfo("Title", "a Tk MessageBox")
#else:
data =  np.loadtxt(file_path,dtype = str)
print('\n')
print('====================================================================================')
print('\n')
print('PROGRAM STARTS ON %s' %today+'  AT %s' %datetime.time(datetime.now()))
NSAM = data[0,1]
sample = NSAM
newpath = os.getcwd() + '/DATA_%s'%sample
if not os.path.exists(newpath):
    os.makedirs(newpath)
# ============================== PARAMETERS ==============================================

nm   = 1.0e-9
b1   = float(data[1,3])*nm
qw1  = float(data[2,3])*nm
b2   = float(data[3,3])*nm
qw2  = float(data[4,3])*nm
b3   = float(data[5,3])*nm
L    = b1+qw1+b2+qw2+b3
N    = int(data[7,1])# No. of points
xx   = 0.0
dx   = L/(N)
x    = np.linspace(0, L, N)
Vcb  = np.zeros(N,dtype = np.float32)
Vvb  = np.zeros(N,dtype = np.float32)

# CONSTANTS
e0          = scipy.constants.e
hbar        = scipy.constants.hbar
m0          = scipy.constants.electron_mass
meV         = 1e-3;
eV2J        = e0
J2eV        = 1./eV2J
Temperature = float(data[6,1])
EgGaAsLT    = 1.519
amplitude   = float(data[9,1])
#compositions
xb1         = float(data[1,2])
xb2         = float(data[3,2])
xb3         = float(data[5,2])
xcomp1 = xb1
xcomp2 = xb2
xcomp3 = xb3
#materials
mb1         = data[1,1]
mqw1        = data[2,1]
mb2         = data[3,1]
mqw2        = data[4,1]
mb3         = data[5,1]

if mb2   == 'AlGaAs':
    Qc   = 0.67
    Qv   = 0.33
elif mb2 == 'AlAs':
    Qc   = 0.70
    Qv   = 0.30


#Energy binding
Eb = float(data[8,1])

# COLORS#
COLORS = ['r','b','sienna','darkcyan','olive','navy','teal','gold','royalblue','m','limegreen']


# Materials
if (len(mb1)==6) and (len(mb2) == 6) and (len(mb3)==6):
    b1m = []
    b2m = []
    b3m = []
    lmb1 = list(mb1)
    lmb2 = list(mb2)
    lmb3 = list(mb3)
    for  ii in range(len(mb1)):
        if ii == 1:
            b1m.append(lmb1[ii]+'(%.2f)'%xb1)
            b2m.append(lmb2[ii]+'(%.2f)'%xb2)
            b3m.append(lmb3[ii]+'(%.2f)'%xb3)
        elif ii == 3:
            b1m.append(lmb1[ii]+'(%.2f)'%float(1-xb1))
            b2m.append(lmb2[ii]+'(%.2f)'%float(1-xb2))
            b3m.append(lmb3[ii]+'(%.2f)'%float(1-xb3))
        else:
            b1m.append(lmb1[ii])
            b2m.append(lmb2[ii])
            b3m.append(lmb3[ii])
    b1m = ''.join(b1m)
    b2m = ''.join(b2m)
    b3m = ''.join(b3m)
elif (len(mb1)==6) and (len(mb2) != 6) and (len(mb3)==6):
    b1m = []
    b3m = []
    lmb1 = list(mb1)
    lmb3 = list(mb3)
    for  ii in range(len(mb1)):
        if ii == 1:
            b1m.append(lmb1[ii]+'(%.2f)'%xb1)
            b3m.append(lmb3[ii]+'(%.2f)'%xb3)
        elif ii == 3:
            b1m.append(lmb1[ii]+'(%.2f)'%float(1-xb1))
            b3m.append(lmb3[ii]+'(%.2f)'%float(1-xb3))
        else:
            b1m.append(lmb1[ii])
            b3m.append(lmb3[ii])
    b1m = ''.join(b1m)
    b2m = mb2
    b3m = ''.join(b3m)


else:
    b1m = mb1
    b2m = mb2
    b3m = mb3

# elements to tabulate
qw1m = mqw1
qw2m = mqw2
db1s  = '<--%.2f nm-->'%(b1/nm)
dqw1s = '<--%.2f nm-->'%(qw1/nm)
db2s  = '<--%.2f nm-->'%(b2/nm)
dqw2s = '<--%.2f nm-->'%(qw2/nm)
db3s  = '<--%.2f nm-->'%(b3/nm)


sp = ' '
cols = 100
print('SAMPLE %s STRUCTURE:\n\n'%NSAM)
print(tabulate([[sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp],
                [b1m ,qw1m,b2m,qw2m, b3m ],
                [sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp]],
                [db1s,dqw1s,db2s,dqw2s,db3s], tablefmt='orgtbl',stralign='center',))
print('\n\n')

# =============================== BAND STRUCTURE =========================================

def STRUCTURE(xx):
    if (mb1==mb2) and (mb1 == mb3):
        if(xb1 == xb2) and (xb1 == xb3):
            if   ( 0.0 < xx <= b1):
                return 1
            elif (  b1 <= xx  <= b1+qw1):
                return 0
            elif (b1+qw1 <= xx <= b1+qw1+b2):
                return 1
            elif (b1+qw1+b2 <= xx <= b1+qw1+b2+qw2):
                return 0
            elif (b1+qw1+b2+qw2 <= xx <= b1+qw1+b2+qw2+b3):
                return 1
            else:
                return 1
        elif(xb1 != xb2) and (xb1 == xb3):
            if   ( 0.0 < xx <= b1):
                return 1
            elif (  b1 <= xx  <= b1+qw1):
                return 0
            elif (b1+qw1 <= xx <= b1+qw1+b2):
                return 2
            elif (b1+qw1+b2 <= xx <= b1+qw1+b2+qw2):
                return 0
            elif (b1+qw1+b2+qw2 <= xx <= b1+qw1+b2+qw2+b3):
                return 1
            else:
                return 1
        elif(xb1 != xb2) and (xb1 != xb3):
            if   ( 0.0 < xx <= b1):
                return 1
            elif (  b1 <= xx  <= b1+qw1):
                return 0
            elif (b1+qw1 <= xx <= b1+qw1+b2):
                return 2
            elif (b1+qw1+b2 <= xx <= b1+qw1+b2+qw2):
                return 0
            elif (b1+qw1+b2+qw2 <= xx <= b1+qw1+b2+qw2+b3):
                return 3
            else:
                return 1
    elif (mb1!=mb2) and (mb1 == mb3):
        if   ( 0.0 < xx <= b1):
            return 1
        elif (  b1 <= xx  <= b1+qw1):
            return 0
        elif (b1+qw1 <= xx <= b1+qw1+b2):
            return 4
        elif (b1+qw1+b2 <= xx <= b1+qw1+b2+qw2):
            return 0
        elif (b1+qw1+b2+qw2 <= xx <= b1+qw1+b2+qw2+b3):
            if(xb1 == xb3):
                return 1
            else:
                return 3
        else:
            return 1



def VCB(xx):
    T = Temperature
    EgGaAs          = (EgGaAsLT-(5.408e-4*T**2/(204+T)))*e0
    EgAlAs          = 3.13*e0
    EgAlGaAs1       = (EgGaAsLT+1.48*xcomp1)*e0
    EgAlGaAs2       = (EgGaAsLT+1.48*xcomp2)*e0
    EgAlGaAs3       = (EgGaAsLT+1.48*xcomp3)*e0
    DeltaEgAlGaAs1  = EgAlGaAs1 - EgGaAs
    DeltaEgAlGaAs2  = EgAlGaAs2 - EgGaAs
    DeltaEgAlGaAs3  = EgAlGaAs3 - EgGaAs
    DeltaEgAlAs     = EgAlAs-EgGaAs
    if   (STRUCTURE(xx)==1):
            return  EgGaAs + DeltaEgAlGaAs1*Qc
    elif (STRUCTURE(xx)==0):
            return  EgGaAs
    elif (STRUCTURE(xx)==2):
            return  EgGaAs + DeltaEgAlGaAs2*Qc
    elif (STRUCTURE(xx)==3):
            return  EgGaAs + DeltaEgAlGaAs3*Qc
    elif (STRUCTURE(xx)==4):
            return  EgGaAs + DeltaEgAlAs*Qc
    else:
        return  EgGaAs + DeltaEgAlGaAs1*Qc

def VVB(xx):
    T = Temperature
    EgGaAs          = (EgGaAsLT-(5.408e-4*T**2/(204+T)))*e0
    EgAlAs          = 3.13*e0
    EgAlGaAs1       = (EgGaAsLT+1.48*xcomp1)*e0
    EgAlGaAs2       = (EgGaAsLT+1.48*xcomp2)*e0
    EgAlGaAs3       = (EgGaAsLT+1.48*xcomp3)*e0
    DeltaEgAlGaAs1  = EgAlGaAs1 - EgGaAs
    DeltaEgAlGaAs2  = EgAlGaAs2 - EgGaAs
    DeltaEgAlGaAs3  = EgAlGaAs3 - EgGaAs
    DeltaEgAlAs     = EgAlAs-EgGaAs
    if   (STRUCTURE(xx)==1):
            return  - DeltaEgAlGaAs1*Qv
    elif (STRUCTURE(xx)==0):
            return  0
    elif (STRUCTURE(xx)==2):
            return  - DeltaEgAlGaAs2*Qv
    elif (STRUCTURE(xx)==3):
            return  - DeltaEgAlGaAs3*Qv
    elif (STRUCTURE(xx)==4):
            return  - DeltaEgAlAs*Qv
    else:
        return  - DeltaEgAlGaAs1*Qv
# ===============================================================================================



# ============================== QUANTUM WELLS POTENTIAL ===================================
for i in tqdm(range(N), desc = 'potential  profile     -', ascii = False,ncols = cols):
    xx = xx + dx
    Vcb[i] =  J2eV*VCB(xx)
    Vvb[i] =  J2eV*VVB(xx)
# ===============================================================================================




# ================================= EFFECTIVE MASS =========================================
# Electron Effective Mass
def effmee(point):
    mAlGaAs1  = (0.067+0.083*xcomp1)*m0
    mAlGaAs2  = (0.067+0.083*xcomp2)*m0
    mAlGaAs3  = (0.067+0.083*xcomp3)*m0
    mGaAs    = 0.067*m0
    mAlAs    = 0.15*m0
    if    (STRUCTURE(point) == 1):
            return mAlGaAs1
    elif  (STRUCTURE(point) == 2):
            return mAlGaAs2
    elif  (STRUCTURE(point) == 0):
            return mGaAs
    elif  (STRUCTURE(point) == 3):
            return mAlGaAs3
    elif  (STRUCTURE(point) == 4):
            return mAlAs
    else:
            return mAlGaAs1

# HEAVY HOLE Effective Mass
def effmhh(point):
    mhhAlGaAs1  = (0.51 + 0.20*xcomp1)*m0
    mhhAlGaAs2  = (0.51 + 0.20*xcomp2)*m0
    mhhAlGaAs3  = (0.51 + 0.20*xcomp3)*m0
    mhhGaAs    = 0.51*m0
    mhhAlAs    = 0.71*m0
    if    (STRUCTURE(point) == 1):
            return mhhAlGaAs1
    elif  (STRUCTURE(point) == 2):
            return mhhAlGaAs2
    elif  (STRUCTURE(point) == 0):
            return mhhGaAs
    elif  (STRUCTURE(point) == 3):
            return mhhAlGaAs3
    elif  (STRUCTURE(point) == 4):
            return mhhAlAs
    else:
            return  mhhAlGaAs1

def effmlh(point):
    mlhAlGaAs1  = (0.082 + 0.078*xcomp1)*m0
    mlhAlGaAs2  = (0.082 + 0.078*xcomp2)*m0
    mlhAlGaAs3  = (0.082 + 0.078*xcomp3)*m0
    mlhGaAs    = 0.082*m0
    mlhAlAs    = 0.16*m0
    if    (STRUCTURE(point) == 1):
            return mlhAlGaAs1
    elif  (STRUCTURE(point) == 2):
            return mlhAlGaAs2
    elif  (STRUCTURE(point) == 0):
            return mlhGaAs
    elif  (STRUCTURE(point) == 3):
            return mlhAlGaAs3
    elif  (STRUCTURE(point) == 4):
            return mlhAlAs
    else:
            return  mlhAlGaAs1
# ===============================================================================================





#%======================== HAMILTONIAN FINITE DIFFERENCE METHOD ============================

#---------------------------------------------------------------------------------------------#
#                                     ELECTRON EIGENVALUES                                    #
#---------------------------------------------------------------------------------------------#
def HAMILTONIAN_e():
    xx       = 0.0
    He       = np.zeros((N,N),dtype = np.float32)
    m_minus  = (effmee(xx) + effmee(xx-dx))/2.0
    m_plus   = (effmee(xx+dx)    + effmee(xx))/2.0
    sn_minus =  -pow(hbar/dx,2)/(2.0*m_minus)
    sn_plus  =  -pow(hbar/dx,2)/(2.0*m_plus)
    bi       =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    He[0,0]  = bi + VCB(xx)
    He[0,1]  = sn_plus
    for ii in tqdm(range(1,N-2), desc = 'electron   hamiltonian -', ascii = False,ncols = cols):
        xx           = xx+dx
        m_minus      = (effmee(xx) + effmee(xx-dx))/2.0
        m_plus       = (effmee(xx+dx)    + effmee(xx))/2.0
        sn_minus     =  -pow(hbar/dx,2)/(2.0*m_minus)
        sn_plus      =  -pow(hbar/dx,2)/(2.0*m_plus)
        bi           =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
        He[ii,ii-1]  = sn_minus
        He[ii,ii]    = bi + VCB(xx)
        He[ii,ii+1]  = sn_plus
    xx           = xx+dx
    m_minus     = (effmee(xx) + effmee(xx-dx))/2.0
    m_plus      = (effmee(xx+dx)    + effmee(xx))/2.0
    sn_minus    =  -pow(hbar/dx,2)/(2*m_minus)
    sn_plus     =  -pow(hbar/dx,2)/(2*m_plus)
    bi          =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    He[N-1,N-1] = bi + VCB(xx)
    He[N-1,N-2] = sn_minus
    [eigvals_e,eigvec_e] = LA.eigh(He)
    return [eigvals_e,eigvec_e]
[eigvals_e,eigvec_e] = HAMILTONIAN_e()
Ee = np.zeros(N,dtype = np.float32)

for m in tqdm(range(0,N), desc = 'electron   energies    -',  ascii = False,ncols = cols):
    Ee[m] = J2eV*eigvals_e[m]
    #sleep(0.00000001)


#---------------------------------------------------------------------------------------------#
#                               HEAVY HOLES EIGENVALUES                                       #
#---------------------------------------------------------------------------------------------#
def HAMILTONIAN_hh():
    xx       = 0.0
    Hhh      = np.zeros((N,N),dtype = np.float32)
    m_minus  = (effmhh(xx) + effmhh(xx-dx))/2.0
    m_plus   = (effmhh(xx+dx)    + effmhh(xx))/2.0
    sn_plus  =  -pow(hbar/dx,2)/(2*m_plus)
    sn_minus =  -pow(hbar/dx,2)/(2*m_minus)
    dn       =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    Hhh[0,0] = dn - VVB(xx)
    Hhh[0,1] =  sn_plus
    for ii in tqdm(range(1,N-2), desc = 'heavy-hole hamiltonian -', ascii = False,ncols = cols):
        xx           = xx+dx

        m_minus      = (effmhh(xx) + effmhh(xx-dx))/2.0
        m_plus       = (effmhh(xx+dx)    + effmhh(xx))/2.0
        sn_plus      =  -pow(hbar/dx,2)/(2*m_plus)
        sn_minus     =  -pow(hbar/dx,2)/(2*m_minus)
        dn           =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
        Hhh[ii,ii-1] = sn_minus
        Hhh[ii,ii]   = dn - VVB(xx)
        Hhh[ii,ii+1] = sn_plus
    xx           = xx+dx
    m_minus      = (effmhh(xx) + effmhh(xx-dx))/2.0
    m_plus       = (effmhh(xx+dx)    + effmhh(xx))/2.0
    sn_plus      =  -pow(hbar/dx,2)/(2*m_plus)
    sn_minus     =  -pow(hbar/dx,2)/(2*m_minus)
    dn           =  0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    Hhh[N-1,N-1] = dn - VVB(xx)
    Hhh[N-1,N-2] = sn_minus
    [eigvals_hh,eigvec_hh] = LA.eigh(Hhh)
    return [eigvals_hh,eigvec_hh]

[eigvals_hh,eigvec_hh] = HAMILTONIAN_hh()
Ehh = np.zeros(N,dtype = np.float32)
for m in tqdm(range(0,N), desc = 'heavy-hole energies    -', ascii = False,ncols = cols):
    Ehh[m] = J2eV*eigvals_hh[m]
    #sleep(0.00000001)


#---------------------------------------------------------------------------------------------#
#                               HEAVY HOLES EIGENVALUES                                       #
#---------------------------------------------------------------------------------------------#
def HAMILTONIAN_lh():
    xx       = 0.0
    Hlh      = np.zeros((N,N),dtype = np.float32)
    m_minus  = (effmlh(xx) + effmlh(xx-dx))/2.0
    m_plus   = (effmlh(xx+dx)    + effmlh(xx))/2.0
    sn_plus   =  -pow(hbar/dx,2)/(2*m_plus)
    sn_minus  =  -pow(hbar/dx,2)/(2*m_minus)
    dn       =   0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    Hlh[0,0] = dn - VVB(xx)
    Hlh[0,1] =  sn_plus
    for ii in tqdm(range(1,N-2), desc = 'light-hole hamiltonian -',ascii = False,ncols = cols):
        xx           = xx+dx
        m_minus      = (effmlh(xx) + effmlh(xx-dx))/2.0
        m_plus       = (effmlh(xx+dx)    + effmlh(xx))/2.0
        sn_plus      =  -pow(hbar/dx,2)/(2*m_plus)
        sn_minus     =  -pow(hbar/dx,2)/(2*m_minus)
        dn           =   0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
        Hlh[ii,ii-1] = sn_minus
        Hlh[ii,ii]   = dn - VVB(xx)
        Hlh[ii,ii+1] = sn_plus
    xx           = xx+dx
    m_minus      =  (effmlh(xx) + effmlh(xx-dx))/2.0
    m_plus       =  (effmlh(xx+dx)    + effmlh(xx))/2.0
    sn_plus      =  -pow(hbar/dx,2)/(2*m_plus)
    sn_minus     =  -pow(hbar/dx,2)/(2*m_minus)
    dn           =   0.5*pow(hbar/dx,2)*((m_plus+m_minus)/(m_plus*m_minus))
    Hlh[N-1,N-1] = dn - VVB(xx)
    Hlh[N-1,N-2] = sn_minus

    [eigvals_lh,eigvec_lh] = LA.eigh(Hlh)
    return [eigvals_lh,eigvec_lh]

[eigvals_lh,eigvec_lh] = HAMILTONIAN_lh()
Elh = np.zeros(N,dtype = np.float32)
for m in tqdm(range(0,N), desc = 'light-hole energies    -', ascii = False,ncols = cols):
    Elh[m] = J2eV*eigvals_lh[m]
    #sleep(0.00000001)




#============================= BAND TO BAND TRANSITIONS ===================================#

ens = input('\n\nNo. Of Transition Energies (>0)?===> ')

if int(ens) == 0:
    print('Error you select 0')
    sys.exit(0)


#if (ks == 'yes'):
HHE   = np.zeros(int(ens),dtype = np.float32)
LHE   = np.zeros(int(ens),dtype = np.float32)
PSIe  = np.zeros((int(ens),N),dtype = np.float32)
PSIhh = np.zeros((int(ens),N),dtype = np.float32)
PSIlh = np.zeros((int(ens),N),dtype = np.float32)

for k in range(1,int(ens)+1):
    HHE[k-1]     = Ee[k]+Ehh[k]-Eb*meV
    LHE[k-1]     = Ee[k]+Elh[k]-Eb*meV 


eprint = np.zeros((2*int(ens),3),dtype = object)
thelist = list(range(0, 2*int(ens)))
cc1 = 0
for ii in thelist[::2]:
    eprint[ii,:] = [str('E'+str(cc1+1)+'-'+'HH'+str(cc1+1)),'%.4f'% HHE[cc1] ,'%.4f'%(1239.4/ HHE[cc1])]
    cc1 = cc1+1

cc1 = 0
for ii in thelist[1::2]:
    eprint[ii,:] = [str('E'+str(cc1+1)+'-'+'LH'+str(cc1+1)),'%.4f'%LHE[cc1] ,'%.4f'%(1239.4/LHE[cc1])]
    cc1 = cc1+1


# PRINT TRANSISITIONS
print('\n')
print(tabulate(eprint,['TRANSITION','ENERGY [eV]','WAVELENGTH'], tablefmt='orgtbl',stralign='center',floatfmt='.4f'))


# ENERGIES VECTOR
eprinte = np.zeros((int(ens),3),dtype = object)
for ii in range(int(ens)):
    eprinte[ii,:] = [str('E'+str(ii+1)+'-> '+'%.4f'%Ee[ii+1]),
                     str('HH'+str(ii+1)+'->'+'%.4f'%-Ehh[ii+1]),
                     str('LH'+str(ii+1)+'->'+'%.4f'%-Elh[ii+1])]

###############################################################################################
#                                                                                             #
#                                 EXPORT ENERGIEs DATA                                        #
#                                                                                             #                          
###############################################################################################


trans_file = open(newpath +'/Transitions-of-sample-%s.txt'%NSAM,'w')
trans_file.write('PROGRAM STARTS ON %s' %today+'  AT %s' %datetime.time(datetime.now()))
trans_file.write('\nTHIS PROGRAM CALCULATE OPTICAL TRANSITIONS OF SAMPLE: '+sample)
trans_file.write('\nTHE SIMPLE STRUCTURE OF THIS SAMPLE:')
trans_file.write('\n\n')
trans_file.write(tabulate(
                [[sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp],
                [b1m ,qw1m,b2m,qw2m, b3m ],
                [sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp],
                [sp,sp,sp,sp,sp]],
                [db1s,dqw1s,db2s,dqw2s,db3s], tablefmt='orgtbl',stralign='center'))
trans_file.write('\n')
trans_file.write('\nEMPLOYED PARAMETERS IN THIS CALCULATIONS: \n')
trans_file.write(tabulate([['TEMPERATURE [K]:' ,Temperature],
                           ['xbarr1:'  ,str('%.2f'%xcomp1)],
                           ['xbarr2:'  ,str('%.2f'%xcomp2)],
                           ['xbarr3:'  ,str('%.2f'%xcomp3)],
                           ['BAND OFFSET:'    ,str(str(Qc)+':'+str(Qv))],
                           ['EXCITON BINDING [eV]:',str('%.2f'%Eb)]], tablefmt='rst',stralign='right'))
#NERGIES
trans_file.write('\n\nENERGIES:\n')
trans_file.write(tabulate(eprinte,['ELECTRON [eV]','HEAVY HOLE [eV]','LIGHT HOLE  [eV]'],tablefmt='orgtbl',stralign='center',floatfmt='.4f'))
trans_file.write('\n')
trans_file.write('\n\nTRANSITIONS:\n')
trans_file.write(tabulate(eprint,['TRANSITION No.','ENERGY    [eV]','WAVELENGTH  [nm]'], tablefmt='orgtbl',stralign='center',floatfmt='.4f'))
end = timeit.default_timer()
if (end - start_time) < 60:
    print(" CPU TIME(s)\t : %.2f"  %(end - start_time))
    trans_file.write('\n\nCPU TIME (s) : \t %.2f'%(end - start_time))
else:
    print(" CPU TIME(min)\t : %.4s" %((end - start_time)/60))
    trans_file.write("\n\nCPU TIME(min) :  %.2f" %((end - start_time)/60))
trans_file.close()



#==================================== EXPORT  DATA =========================================
#Para no exportar todos los datos calcular la el rango adecuado

xmindata =  int((b1 -30*nm)/dx)
xmaxdata =  int((b1+qw1+b2+qw2 +30*nm)/dx)
ran      = xmaxdata-xmindata
PSIe  = np.zeros((int(ens),ran),dtype = np.float32)
PSIhh = np.zeros((int(ens),ran),dtype = np.float32)
PSIlh = np.zeros((int(ens),ran),dtype = np.float32)


export = input('\nDo you want export data? [Y/n] ===>')
if (export == 'y') or (export == 'Y'):
    absolute =  input('Do you prefer absolute wave functions? [Y/n] ===> ')
    if (absolute == 'y'):
        for k in tqdm(range(1,int(ens)+1), desc = 'exp.wave   functions   -', ascii = False,ncols = cols):
            PSIe[k-1,:]  = amplitude*abs(eigvec_e[xmindata:xmaxdata,k])**2+Ee[k]
            PSIhh[k-1,:] = amplitude*abs(eigvec_hh[xmindata:xmaxdata,k])**2-Ehh[k]
            PSIlh[k-1,:] = amplitude*abs(eigvec_lh[xmindata:xmaxdata,k])**2-Elh[k]
    else:
        for k in tqdm(range(1,int(ens)+1),desc = 'exp.wave   functions   -', ascii = False,ncols = cols):
            PSIe[k-1,:]  = amplitude*eigvec_e[xmindata:xmaxdata,k]+Ee[k]
            PSIhh[k-1,:] = amplitude*eigvec_hh[xmindata:xmaxdata,k]-Ehh[k]
            PSIlh[k-1,:] = amplitude*eigvec_lh[xmindata:xmaxdata,k]-Elh[k]
            #sleep(0.00000001)
    # POTENTIAL PROFILES
    with open(newpath +"/Potential_Profile_%s"%NSAM + ".dat", "w") as out_file:
           for i in tqdm(range(xmindata,xmaxdata),desc = 'exp.potential  profile -', ascii = False,ncols = cols):
               out_string = ""
               out_string += str(x[i]/nm)
               out_string += ","  + str(Vcb[i])
               out_string += ","  + str(Vvb[i])
               out_string += "\n"
               out_file.write(out_string)
    # Wave Function electron
    with open(newpath +"/PSIelectron_data_%s"%NSAM + ".dat", "w") as out_file:
        for i in tqdm(range(ran),desc = 'exp.electron   wavefun -', ascii = False,ncols = cols):
            out_string = ""
            out_string += str(x[xmindata+i]/nm)
            for j in range(int(ens)):
                out_string += ","  + str(PSIe[j,i])
            out_string += "\n"
            out_file.write(out_string)
    # Wave Function heavy hole
    with open(newpath +"/PSIheavyhole_data_%s"%NSAM + ".dat", "w") as out_file:
        for i in tqdm(range(ran),desc = 'exp.heavy-hole wavefun -', ascii = False,ncols = cols):
            out_string = ""
            out_string += str(x[xmindata+i]/nm)
            for j in range(int(ens)):
                out_string += ","  + str(PSIhh[j,i])
            out_string += "\n"
            out_file.write(out_string)
    # Wave Function light hole
    with open(newpath +"/PSIlighthole_data_%s"%NSAM + ".dat", "w") as out_file:
        for i in tqdm(range(ran),desc = 'exp.light-hole wavefun -', ascii = False,ncols = cols):
            out_string = ""
            out_string += str(x[xmindata+i]/nm)
            for j in range(int(ens)):
                out_string += ","  + str(PSIlh[j,i])
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






# ==================================== PLOT ================================================


aplot = input('\nDo you want plot the wave fuctions? [Y/n] ===> ')
if (aplot == 'y') or (aplot == 'Y'):
    absolute =  input('Do you prefer absolute wave functions? [Y/n] ===> ')
    if (aplot == 'y') or (aplot == 'Y'):
        for k in tqdm(range(1,int(ens)+1),desc = 'Generating Wave Functions to plot      ',ascii = False,ncols = cols):
            PSIe[k-1,:]  = amplitude*abs(eigvec_e[xmindata:xmaxdata,k])**2+Ee[k]
            PSIhh[k-1,:] = amplitude*abs(eigvec_hh[xmindata:xmaxdata,k])**2-Ehh[k]
            PSIlh[k-1,:] = amplitude*abs(eigvec_lh[xmindata:xmaxdata,k])**2-Elh[k]
        if (max(Vcb) < max(PSIe[int(ens)-1,:])):
            ymaxe  = max(PSIe[int(ens)-1,:])+0.1
        else: 
            ymaxe  = max(Vcb) + 0.1
            
        xmin = b1/nm-30
        xmax = (b1+qw1+b2+qw2)/nm+30
        ymine =  min(Vcb[:])-0.02
        yminh =  min(Vvb[:])-0.02
        ymaxh  = max(PSIhh[1,:])+0.02
    else:
        for k in tqdm(range(1,int(ens)+1),desc = 'Generating Wave Functions to plot      ',ascii = False,ncols = cols):
            PSIe[k-1,:]  = amplitude*eigvec_e[xmindata:xmaxdata,k]+Ee[k]
            PSIhh[k-1,:] = amplitude*eigvec_hh[xmindata:xmaxdata,k]-Ehh[k]
            PSIlh[k-1,:] = amplitude*eigvec_lh[xmindata:xmaxdata,k]-Elh[k]
        if (max(Vcb) < max(PSIe[int(ens)-1,:])):
            ymaxe  = max(PSIe[int(ens)-1,:])+0.02
        else: 
            ymaxe  = max(Vcb) + 0.05
        xmin = b1/nm-30
        xmax = (b1+qw1+b2+qw2)/nm+30
        ymaxe  = max(PSIe[int(ens)-1,:])+0.02
        ymine =  min(PSIe[0,:])-0.02
        yminh =  min(PSIlh[int(ens)-1,:])-0.02
        ymaxh  = max(PSIhh[1,:])+0.02
    # POTENTIAL PROFILES
    fig, (ax,ax2) = plt.subplots(2, 1,figsize=(5,7), sharex=True)
    plt.subplots_adjust(left=0.15, right=0.9, bottom = 0.1,top=0.9,hspace=0.1)
    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    #================================== Electron =======================================
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymine,ymaxe)
    ax.tick_params(bottom="off", labelbottom='off')
    ax.spines['bottom'].set_visible(False)
    ax.plot(x*1e9,Vcb,'-k')
    for p in range(int(ens)):
        ax.plot(x[xmindata:xmaxdata]/nm,PSIe[p,:],'-',label='$\psi e$'+str(int(p+1)))
    legend = ax.legend(loc='upper right',prop={'size': 8},ncol=1)
    ax.minorticks_on()
    ##ax2.tick_params(axis='y', which='minor', right='off')
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.tick_params(axis='y', length=4,which='minor', direction='in')
    ax.tick_params(axis='y', length=8,which='major', direction='in')
    #=============================== Heavy and light holes ==============================
    ax2.set_xlim(xmin,xmax)
    ax2.set_ylim(yminh,ymaxh)
    ax2.plot(x/nm,Vvb,'-k')
    for p in range(int(ens)):
        ax2.plot(x[xmindata:xmaxdata]/nm,PSIhh[p,:],'--',label='$\psi HH$'+str(int(p+1)))
    for p in range(int(ens)):
        ax2.plot(x[xmindata:xmaxdata]/nm,PSIlh[p,:],':',label='$\psi LH$'+str(int(p+1)))
    legend = ax2.legend(loc='lower right',prop={'size': 8},ncol=2)
    ax2.minorticks_on()
    ax2.set_xlabel(r'z[nm]', fontsize=18)
    ax2.set_ylabel(r'Energy[eV]',x=-0.1,y=0.8, ha='left', fontsize=18)
    ax2.tick_params(axis='x', length=4,which='minor', direction='in')
    ax2.tick_params(axis='x', length=8,which='major', direction='in')
    ax2.tick_params(axis='y', length=4,which='minor', direction='in')
    ax2.tick_params(axis='y', length=8,which='major', direction='in')
    # # draw skiwers
    d  = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (0, 0), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (0, 0), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - 0, 1 + 0), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - 0, 1 + 0), **kwargs)  # bottom-right diagonal
    plt.show()
    end = timeit.default_timer()
    if (end - start_time) < 60:
        print("CPU TIME(s) :\t %.2f"  %(end - start_time))
    else:
        print("CPU TIME(min) : \t %.2f" %((end - start_time)/60))

else:
     print('ok')
     end = timeit.default_timer()
     if (end - start_time) < 60:
         print("CPU TIME(s) :\t %.2f"  %(end - start_time))
     else:
        print("CPU TIME(min) : \t %.2f" %((end - start_time)/60))
