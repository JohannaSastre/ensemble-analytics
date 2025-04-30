import os
import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg

#Modeule to install
#pip install streamlit 
#pip install streamlit-echarts 
#pip install streamlit-option-menu 
#pip install matplotlib 
#pip install openpyxl

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

# Add 'GitHub' to the pages list
pages = ["Home", "Pre-Drilling", "While-Drilling", "Post-Drilling"]

# Correct the logo path
logo_path = "C:\\Users\\johsas\\OneDrive - Aker BP\\python_projects\\Project_PlotDrill\\python\\Akerbp.svg"

# Check if the file exists
if not os.path.exists(logo_path):
    st.error("Logo file not found. Please check the file path.")
else:
    # Display the logo (SVG supported in newer Streamlit versions)
    st.image(logo_path, width=150, caption="Aker BP Logo")

urls = {"GitHub": "https://github.com/gabrieltempass/streamlit-navigation-bar"}  # Keep GitHub in urls
styles = {
    "nav": {"background-color": "royalblue", "justify-content": "left"},
    "img": {"padding-right": "14px"},
    "span": {"color": "white", "padding": "14px"},
    "active": {"background-color": "white", "color": "var(--text-color)", "font-weight": "normal", "padding": "14px"},
}
options = {"show_menu": False, "show_sidebar": False}

page = st_navbar(pages, logo_path=logo_path, urls=urls, styles=styles, options=options)

functions = {
    "Home": pg.show_home,
    "Install": pg.show_install,
    "User Guide": pg.show_user_guide,
    # Add a placeholder for GitHub
    "GitHub": lambda: st.write("Visit the [GitHub repository](https://github.com/gabrieltempass/streamlit-navigation-bar)."),
}
go_to = functions.get(page)
if go_to:
    go_to()