import os
import streamlit as st
from app_utils.structure import Structure,model,calculation,energies,down_results
from app_plots.plots import Plots
import cqws.solver_qws as solver



st.set_page_config(page_title="Solution of 1D Schrodinger equation in Quantum Wells",
                   layout="wide",
                    page_icon='icons/logo_iico_azul.png')

col1, col2 = st.columns([3, 1]) # specify the width ratios of the columns
with col1:
    st.title(":blue[Solution of 1D Schrodinger equation in III-V Quantum Wells]")
with col2:
    st.image('icons/logo_iico_azul.png',use_column_width=True)

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
    
    
colsp = st.columns((1,1.5))

structure = Structure()
with colsp[0]:
    # with st.expander("Structure Parameters"):
        parameters=structure.parameters()
        NoQws = st.number_input("Number of Layers",value=3, min_value=3, max_value=13)
        layerss = structure.structure_layers(NoQws)
        with st.expander("Table of total layers"):
            table_layers=structure.table_layers()
            
with colsp[1]:    
    # with st.expander("Results of the structure profile and its wave functions"):
        model = model(structure)
        Plots(model,structure).plot_profile()
        if st.button('Run',use_container_width=True):
            calc = calculation(model)
            Plots(model,structure).plot_calculations(calc)
            with st.expander("Table of Total Subband Energies"):
                table_energies=energies(calc)
                st.dataframe(table_energies,use_container_width=True)
                down_results = down_results(calc)
                subcol1,subcol2,subcol3,subcol4= st.columns(4)
                with subcol1:
                    st.download_button(
                        label="Download data as CSV",
                        data=table_energies.to_csv().encode('utf-8'),
                        file_name='energies.csv',
                        mime='text/csv',
                    )
                with subcol2:
                    st.download_button( label="Download E-WF as CSV",
                        data=down_results.dfen,
                        file_name='Electron_wf.csv',
                        mime='text/csv',)
                with subcol3:
                    st.download_button( label="Download HH-WF as CSV",
                        data=down_results.dfhh,
                        file_name='Heavy-Hole_wf.csv',
                        mime='text/csv',)
                with subcol4:
                    st.download_button( label="Download LH-WF as CSV",
                        data=down_results.dfhh,
                        file_name='Light-Hole_wf.csv',
                        mime='text/csv',)
                
                
                
            
    
# with colsp[0]:    
#     with st.expander("Table of total layers"):
#         layers=structure.table_layers()
    

    
