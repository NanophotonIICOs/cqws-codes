#From https://github.com/MannLabs/OmicLearn/blob/master/omiclearn/utils/ui_helper.py
import plotly
import os, sys
import base64
import sklearn
import numpy as np
import pandas as pd
import streamlit as st


# Widget for recording
def make_recording_widget(f, widget_values):
    """
    Return a function that wraps a streamlit widget and records the
    widget's values to a global dictionary.
    """

    def wrapper(label, *args, **kwargs):
        widget_value = f(label, *args, **kwargs)
        widget_values[label] = widget_value
        return widget_value

    return wrapper

_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
_parent_directory = os.path.dirname(_this_directory)

# Object for dict
class objdict(dict):
    """
    Objdict class to conveniently store a state
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


# Main components
def main_components():
    """
    Expose external CSS and create & return widgets
    """

    # Fundemental elements
    widget_values = objdict()
    record_widgets = objdict()

    # Sidebar widgets
    sidebar_elements = {
        "button_": st.sidebar.button,
        "slider_": st.sidebar.slider,
        "number_input_": st.sidebar.number_input,
        "selectbox_": st.sidebar.selectbox,
        "multiselect": st.multiselect,
    }
    for sidebar_key, sidebar_value in sidebar_elements.items():
        record_widgets[sidebar_key] = make_recording_widget(
            sidebar_value, widget_values
        )

    return widget_values, record_widgets


# Generate sidebar elements
def generate_sidebar_elements(state, icon, report, record_widgets):
    slider_ = record_widgets.slider_
    selectbox_ = record_widgets.selectbox_
    number_input_ = record_widgets.number_input_

    # Sidebar -- Random State
    state["random_state"] = slider_(
        "Random State:", min_value=0, max_value=99, value=23
    )

    return state


# Prepare system report
def get_system_report():
    """
    Returns the package versions
    """
    report = {}
    report["omic_learn_version"] = "v1.3.1"
    report["python_version"] = sys.version[:5]
    report["pandas_version"] = pd.__version__
    report["numpy_version"] = np.version.version
    report["sklearn_version"] = sklearn.__version__
    report["plotly_version"] = plotly.__version__

    return report