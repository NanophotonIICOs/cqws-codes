import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cqws.solver_qws as solver
import streamlit as st


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

class Plots:
    def __init__(self,model,structure) -> None:
        nm = 1e-9
        self.cb = model.cb/q
        self.vb = model.vb/q
        self.z  = model.xaxis/nm
        #self.subs_width =  structure.total_layers[0][0]
        #self.xr = self.subs_width-1
        
    def plot_profile(self):
        cut_interval = [0,np.min(self.cb)]
        fig = make_subplots(
                                rows=2,
                                cols=1,
                                vertical_spacing=0.05,
                                shared_xaxes=True,
                            )
        
        
        fig.append_trace(go.Scatter(x=self.z,y=self.cb,name="CB"), row=1, col=1,)
        fig.append_trace(go.Scatter(x=self.z,y=self.vb,name="VB"), row=2, col=1,)
        fig.update_yaxes(range=[cut_interval[1], max(self.cb) * 1.01], row=1, col=1)
        fig.update_yaxes(range=[min(self.vb), cut_interval[0]], row=2, col=1)
        #fig.update_xaxes(range=[self.xr, max(self.z)])
        fig.update_layout(yaxis=dict(title="CB-edge (eV)"),
                          yaxis2=dict(title="VB-edge (eV)"),
                          xaxis2=dict(title="Growth Direction (nm)")
                          )
        
        
        st.plotly_chart(fig, use_container_width=True)
        
        
    
    def plot_calculations(self,results):
        
        wf_hh = results.psihh
        wf_lh = results.psilh
        
        cut_interval = [0,np.min(self.cb)-0.0]
        fig = make_subplots(
                                rows=2,
                                cols=1,
                                vertical_spacing=0.05,
                                shared_xaxes=True,
                            )
        
        
        fig.append_trace(go.Scatter(x=self.z,y=self.cb,name="CB"), row=1, col=1,)
        fig.append_trace(go.Scatter(x=self.z,y=self.vb,name="VB"), row=2, col=1,)
        
        for i in range(results.subbands):
            fig.append_trace(
                            go.Scatter(x=self.z,y=10*results.psie[:,i]+results.Ee[i],
                                       name=f'e-{i}'
                                       ), row=1, col=1,)
        
        for i in range(results.subbands):
            fig.append_trace(
                            go.Scatter(x=self.z,y=10*results.psihh[:,i]-results.Ehh[i],
                                       name=f'hh-{i}'
                                       ), row=2, col=1,)
            fig.append_trace(
                            go.Scatter(x=self.z,y=10*results.psilh[:,i]-results.Elh[i],
                                       name=f'hh-{i}'
                                       ), row=2, col=1,)
        
        #fig.update_yaxes(range=[cut_interval[1], max(self.cb) * 1.02], row=1, col=1)
        #fig.update_yaxes(range=[min(self.vb), cut_interval[0]+0.05], row=2, col=1)
        #fig.update_xaxes(range=[self.xr, max(self.z)])
        fig.update_layout(yaxis=dict(title="CB-edge (eV)"),
                          yaxis2=dict(title="VB-edge (eV)"),
                          xaxis2=dict(title="Growth Direction (nm)")
                          )
        
        
        st.plotly_chart(fig, use_container_width=True)

        
