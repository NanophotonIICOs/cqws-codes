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


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#7f0000', '#008b8b', '#a9a9a9', '#8fbc8f', '#483d8b',
          '#b8860b', '#2f4f4f', '#00ced1', '#ff1493', '#228b22',
          '#6b8e23', '#d2b48c', '#696969', '#dc143c', '#00ffff',
          '#7b68ee', '#00fa9a', '#ffd700', '#f0e68c', '#ff00ff',
          '#ffdab9', '#f5f5dc', '#000080', '#add8e6', '#f08080',
          '#90ee90', '#d3d3d3', '#ffb6c1', '#ffffe0', '#ffa07a',
          '#20b2aa', '#87cefa', '#778899', '#32cd32', '#b0c4de',
          '#ffff00', '#00ff00', '#800000', '#00ff7f', '#8b008b',
          '#ff4500', '#fa8072', '#00fa9a', '#4682b4', '#e9967a',
          '#da70d6', '#d8bfd8', '#ff6347', '#40e0d0', '#87ceeb',
          '#fffafa', '#1e90ff', '#b22222', '#6a5acd', '#f4a460',
          '#48d1cc', '#c71585', '#191970', '#afeeee', '#bc8f8f',
          '#ff69b4', '#cd5c5c', '#eee8aa', '#9370db', '#00ffff']






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
        #fig.update_yaxes(range=[cut_interval[1], max(self.cb) * 1.01], row=1, col=1)
        #fig.update_yaxes(range=[min(self.vb), cut_interval[0]], row=2, col=1)
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
        
        
        fig.append_trace(go.Scatter(x=self.z,y=self.cb,name="CB",line=dict(color="gray"),showlegend=False), row=1, col=1,)
        fig.append_trace(go.Scatter(x=self.z,y=self.vb,name="VB",line=dict(color="gray"),showlegend=False), row=2, col=1,)
        
        for i in range(results.subbands):
            fig.append_trace(
                            go.Scatter(x=self.z,y=10*results.psie[:,i]+results.Ee[i],
                                       name=rf"e-{i}",
                                       line=dict(color=colors[i])
                                       ), row=1, col=1,)
        
        for i in range(results.subbands):
            fig.append_trace(
                            go.Scatter(x=self.z,y=10*results.psihh[:,i]-results.Ehh[i],
                                       name=f'hh-{i}',
                                       line=dict(color=colors[i],dash='dash')
                                       ), row=2, col=1,)
            fig.append_trace(
                            go.Scatter(x=self.z,y=10*results.psilh[:,i]-results.Elh[i],
                                       name=f'hh-{i}',
                                       line=dict(color=colors[i],dash='dot')
                                       ), row=2, col=1,)
        
        #fig.update_yaxes(range=[cut_interval[1], max(self.cb) * 1.02], row=1, col=1)
        #fig.update_yaxes(range=[min(self.vb), cut_interval[0]+0.05], row=2, col=1)
        #fig.update_xaxes(range=[self.xr, max(self.z)])
        fig.update_layout(xaxis=dict(title="",showgrid=False),
                          yaxis=dict(title="CB-edge (eV)",showgrid=True,showline=True,linewidth=1, linecolor='black',mirror=True),
                          yaxis2=dict(title="VB-edge (eV)",showgrid=True,showline=True,linewidth=1, linecolor='black',mirror=True),
                          xaxis2=dict(title="Growth Direction (nm)",showgrid=False,showline=True,linewidth=1, linecolor='black',),
                          font_family="Serif", font_size=15, 
                          margin_l=5, margin_t=5, margin_b=5, margin_r=5,
                          legend=dict(
        x=1.02, # ajusta la posición horizontal de la leyenda
        y=1, # ajusta la posición vertical de la leyenda
        bordercolor='black', # ajusta el color del borde de la leyenda
        borderwidth=1, # ajusta el ancho del borde de la leyenda
        bgcolor='white' # ajusta el color de fondo de la leyenda
    )
                          )
        # fig.update_xaxes(showgrid=False,showline=True, linewidth=2, linecolor='black')
        # fig.update_yaxes(showgrid=False,showline=True, linewidth=2, linecolor='black')

        
        
        st.plotly_chart(fig, use_container_width=True,kwargs={"include_mathjax":"cdn"})
        # st.components.v1.html(fig.to_html(include_mathjax='cdn'))
        
