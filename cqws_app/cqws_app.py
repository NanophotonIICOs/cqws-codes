import os
import streamlit as st
from app_utils.structure import Structure,model,calculation
from app_plots.plots import Plots



st.set_page_config(page_title="Solution of 1D Schrodinger equation in Quantum Wells",layout="wide")
st.title("Solution of 1D Schrodinger equation in III-V Quantum Wells")
st.write("")
st.markdown(
    """
   
"""
)



with st.expander("About this app"):

    st.write("")

    st.markdown(
        """
   
    """
    )

    st.write("")

    st.markdown(
        """
   
    """
    )

    st.write("")
    
    
colsp = st.columns(2)


structure = Structure()
with colsp[0]:
    # with st.expander("Structure Parameters"):
        parameters=structure.parameters()
        NoQws = st.number_input("Number of Layers",value=3, min_value=3, max_value=13)
        layerss = structure.structure_layers(NoQws)
        with st.expander("Table of total layers"):
            layers=structure.table_layers()

with colsp[1]:    
    with st.expander("Structure profile"):
        model = model(structure)
        Plots(model,structure).plot_profile()
    with st.expander("Run Calculation"):
        if st.button('Run'):
            calc = calculation(model)
            Plots(model,structure).plot_calculations(calc)
        
        
    
# with colsp[0]:    
#     with st.expander("Table of total layers"):
#         layers=structure.table_layers()
    

    

