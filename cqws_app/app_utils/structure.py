import numpy as np
import cqws.solver_qws as solver
from cqws.tools import plotstyle
import streamlit as st
import pandas as pd


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
                x = st.number_input("Al concentration",value=0.15, min_value=0.1, max_value=0.4,step=0.01, key=f"al_concentration_{layer_index}")
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
            self.grid_factor = st.number_input(r"$\Delta z$ (nm)",min_value=0.01,max_value=1.00,value=0.1,step=0.1)
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
    Structure = model
    results   = solver.Solver(model).QuantumSolutions(absolute =True,Print=False)
    return results
