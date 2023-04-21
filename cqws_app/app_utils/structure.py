import numpy as np
import streamlit as st
import pandas as pd
import cqws.solver_qws as solver
from collections import namedtuple
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
nm=1e-9
J2eV=1/q #Joules to eV
eV2J=1*q #eV to Joules



def layer(layer_index):
    cols = st.columns(4,gap="small")
    with cols[0]:
        layer_width = st.number_input(f"Layer Width {layer_index+1} (nm)",value=10.0, min_value=0.1, max_value=300.0,step=0.1, key=f"width_{layer_index}")
    with cols[1]:
        layer_material = st.radio("Layer material",("GaAs","AlAs","AlGaAs"), key=f"material_{layer_index}")
    with cols[2]:
        layer_type = st.radio("Layer type",("Barrier","QW"), key=f"type_{layer_index}")
        if layer_material=="AlGaAs":
            with cols[3]:
                x = st.number_input("Al concentration",value=0.15, min_value=0.1, max_value=0.5,step=0.01, key=f"al_concentration_{layer_index}")
        else:
            x=0
    return [layer_width,layer_material,x,layer_type]

def substrate():
    cols0 = st.columns(2)
    with cols0[0]:
        subs_material = st.radio("Substrate",("GaAs","AlAs"),) 
    with cols0[1]:
        width_subs = st.number_input("Width (nm)",value=100, min_value=10, max_value=1000)
    return [width_subs,subs_material,0,"Substrate"]
    
def create_structure(NoLayers):
    layer_list = []  # Lista para almacenar los resultados de la función layer()
    with st.expander("Structure layers"):
        # subs=substrate()
        # layer_list.append(subs)
        for i in range(NoLayers):
            layer_result = layer(i)
            layer_list.append(layer_result)
            #st.write(f"Layer {i+1}: {layer_result}")
    
    return layer_list


class AttrDict(dict):
    """turns a dictionary into an object with attribute style lookups"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Structure:
    def __init__(self) -> None:
         self.T = None
         self.Qc = None
         self.Qv = None
         self.subbands = None
         self.total_layers= None   
         self.grid_factor = None 
         self.total_layers= None
         self.df = None
         self.name = None
         self.hhb = None
         self.lhb = None
         self.grid_factor = None
         
    def structure_layers(self,NoLayers):
        self.total_layers= create_structure(NoLayers)
        return self.total_layers
        
    def parameters(self):
        
        col01,col02 = st.columns(2)
        with col01:
            self.name = st.text_input("Structure name",value="Structure")
        with col02:
            self.T = st.number_input("Temperature (K)",min_value=1,max_value=330,value=30)
        st.markdown("Band Offset ratios")
        colqc1,colqc2 = st.columns(2)
        with colqc1:
            self.Qc = st.number_input(r"$Q_c$",min_value=0.1,max_value=0.99,value=0.65,step=0.01)
        with colqc2:
            self.Qv = st.number_input(r"$Q_v$",min_value=0.1,max_value=0.99,value=0.35,step=0.01)
        st.markdown("Calculation Parameters")
        colp1,colp2 = st.columns(2)
        with colp1:
            self.subbands = st.number_input("No. of calculate subbands",min_value=1,max_value=10,value=1)
        with colp2:
            self.grid_factor = st.number_input(r"$\Delta z$ (nm)",min_value=0.1,max_value=1.00,value=0.1,step=0.1)
        st.markdown("Ground Bound Exciton Parameters")
        colb1,colb2=st.columns(2)
        with colb1:
            self.hhb = st.number_input("$HH$ binding (meV)",min_value=0.1,max_value=20.0,value=5.0,step=0.1)*1e-3
        with colb2:
            self.lhb = st.number_input("$LH$ binding (meV)",min_value=0.1,max_value=20.0,value=5.0,step=0.1)*1e-3
         
    def table_layers(self):
        self.df = pd.DataFrame(self.total_layers,columns=["Layer Width (nm)","Material","Al concentration","Type"])
        st.dataframe(self.df)
        
        


def model(structure):
    class Make_Structure(object):pass
    s = Make_Structure( )
    s.T = structure.T
    s.Qc = structure.Qc
    s.Qv = structure.Qv
    s.structure_name = structure.name
    s.HHBinding = structure.hhb
    s.LHBinding =structure.lhb    
    s.gridfactor = structure.grid_factor
    s.Fapp = 0
    s.subbands = structure.subbands
    s.material = structure.total_layers
    model_return = solver.StructureFrom(s)
    return model_return

def calculation(model):    
    results = solver.Solver(model).QuantumSolutions(absolute =True,Print=False)
    return results


def energies(results):
    eigen = results.Energies.T
    dfen  = pd.DataFrame(eigen,columns=["Electron (eV)","Heavy-Hole (eV)","Light-Hole (eV)"])
    return dfen.round(6)

def down_results(results):
    down = namedtuple('down',['dfen','dfhh','dflh'])
    
    en = results.psie
    hh = results.psihh
    lh = results.psilh
    xaxis = results.xaxis/nm
    cb =     results.cb
    vb =  results.vb
    subbands =  results.subbands
    ene_array = np.zeros((len(xaxis),3+subbands))
    enhh_array = enlh_array = ene_array
    ene_array[:,0] = enhh_array[:,0] = enlh_array[:,0] = xaxis
    ene_array[:,1] = enhh_array[:,1] = enlh_array[:,1] =  cb
    ene_array[:,2] = enhh_array[:,2] = enlh_array[:,2] = vb
    
    head_principal = ['z','cb','vb']
    for i in range(subbands):
        ene_array[:,i+3]=en[:,i]
        enhh_array[:,i+3]=hh[:,i]
        enlh_array[:,i+3]=lh[:,i]

    heade  = ['z','cb','vb']
    headhh = ['z','cb','vb']
    headlh = ['z','cb','vb']
    for i in range(subbands):
        heade.append(f"psie-{i}")
        headhh.append(f"psihh-{i}")
        headlh.append(f"psilh-{i}")
    dfen = pd.DataFrame(ene_array,columns=heade).to_csv()
    dfhh = pd.DataFrame(enhh_array,columns=headhh).to_csv()
    dflh = pd.DataFrame(enlh_array,columns=headlh).to_csv()
    
    return down(dfen,dfhh,dflh)