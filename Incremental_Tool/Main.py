
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import streamlit as st
import pandas as pd
import numpy as np
import operator

import matplotlib.pyplot as plt
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
import datetime as dt
from datetime import datetime

import smbclient
import smbclient.path
import smbprotocol.exceptions

from streamlit_option_menu import option_menu
from matplotlib.gridspec import GridSpec

import streamlit as st
from streamlit_option_menu import option_menu

#########################for plotting###################
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots  # Ensure this is included

import sys
sys.path.append("style.py")
#from style import init_page

# Initialize the page with custom styles
#   page_title="Incremental Value",
#    page_icon=":chart_with_upwards_trend:",
#    layout="wide",
#)

st.set_page_config(
    page_title="Incremental Value",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Apply the styles
#st.markdown(apply_styles(), unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Add your custom styles here */
    body {
        background-color: #F2F2F2;
        color: #1A1A1A;
        font-family: "Segoe UI", "Arial", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Add the Aker BP logo to the top-left side
st.sidebar.image("logo.png", width=150)  # Replace with the correct path to your logo

########################################## Config

def sort_key(s):
    import re
    match = re.search(r'\d+', s)
    if match:
        return (0, int(match.group()))
    else:
        return (1, s)

########################################## Sidebar option menu
with st.sidebar:
    st.title("Incremental Ensemble Analysis")
    
    selected = option_menu(
        menu_title="",           # Title of the sidebar
        options=["About","Upload Data", "Filter cases","Sales Conversions", "Boxplots","Analysis","Crossplot","Case selection","Waterfall","Data"
                ],  # Menu options with emojis
        icons=["info-circle", "capslock", "graph-down", "bar-chart-steps","arrows-expand-vertical","diagram-3","currency-dollar","bullseye","bounding-box","clipboard-data", "gear"],  # Optional icons from Bootstrap
        default_index=0,                  # Default selected option
        )

    if 'dict_incremental_mapping' in st.session_state:
        
        incremental_mapping = st.session_state['dict_incremental_mapping']
      
        
        st.header("Filter cases")
        
        df_filters = pd.DataFrame(index = st.session_state['case_groups'].columns,columns = ['Filter']).fillna(False)
        df_filters = st.data_editor(df_filters ,use_container_width = True)
        
        
        df_filtered_cases = st.session_state['case_groups'].copy()
        for group in df_filters.index:
            if df_filters.loc[group,'Filter'] == False:
                df_filtered_cases.drop(columns = group,inplace=True)
        df_filtered_cases = df_filtered_cases[df_filtered_cases.product(axis=1) == 1]
        filtered_cases = list(df_filtered_cases.index)
   
        
        if df_filters.empty:
            st.error("No case filters defined. All cases displayed.")
        else:    
            with st.expander('Currently Displayed Cases:'):
                st.write(filtered_cases)
        
from pathlib import Path

# Point directly to the markdown file in the current directory
ABOUT = Path("about.md")

if selected == "About":
    #st.markdown("<h3 style='text-align: left; font-weight: bold;'>Incremental Dashboard</h3>", unsafe_allow_html=True)
    try:
        with open("about.md", "r", encoding="utf-8") as file:
            about_content = file.read()
            st.markdown(about_content, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("The 'about.md' file is missing. Please ensure it's in the same folder as your script.")

########################################## Main menus
if selected == "Data":
    c1,c2 = st.columns(2)
    with c1:
        if st.checkbox("Show Base data:",value=False):
            if 'data_dict_base' in st.session_state:
                st.json(st.session_state['data_dict_base'],expanded=False)
    with c2:
        if st.checkbox("Show Project data:",value=False):
            if 'data_dict_project' in st.session_state:
                st.json(st.session_state['data_dict_project'],expanded=False)

    
if selected == "Filter cases":
    

    if 'case_groups' in st.session_state:
        case_groups = st.session_state['case_groups']
        
        c1,c2,c3,c4 = st.columns([3,1,5,10])
        
        with c1:
            new_group_name = st.text_input("Add new name:")
        
            if st.button('Add',use_container_width=True):
                case_groups[new_group_name] = False
        with c3:
            file1 = st.file_uploader('Upload pre-defined groups', type=['xlsx'])
            if file1:
                df = pd.read_excel(file1,index_col=0)
    
        st.title('Case groups')
        editable_df = st.data_editor(st.session_state['case_groups'],use_container_width = True, height = 1000)
        st.session_state['case_groups'] = editable_df
    
    
if selected == "Upload Data":
    
    if 'well_groups' not in st.session_state:
        st.session_state['well_groups'] = pd.DataFrame()
    if 'region_groups' not in st.session_state:
        st.session_state['region_groups'] = pd.DataFrame()
    
    col1, col2,col3,col4,col5 = st.columns([1,1,1,1,1])
    col1_, col2_,col3_,col4_,col5_ = st.columns([12,1,12,1,12])
    
    with col1:
        st.markdown("<h2 style='font-size:28px;'>Ensemble Base</h2>", unsafe_allow_html=True)
        file = st.file_uploader(key='1',label='', type=['pkl'])
        if file:
            data_dict_base = pd.read_pickle(file)
            st.session_state['data_dict_base'] =data_dict_base

    with col2:        
        
        if 'data_dict_base' in st.session_state:
            st.markdown("<h2 style='font-size:28px;'>Ensemble Project</h2>", unsafe_allow_html=True)
            file2 = st.file_uploader(key='2', label='', type=['pkl'])
            if file2:
                data_dict_project = pd.read_pickle(file2)
                st.session_state['data_dict_project'] = data_dict_project

    with col3:          
        if 'data_dict_project' in st.session_state:
            st.markdown("<h2 style='font-size:28px;'>Incremental Mapping</h2>", unsafe_allow_html=True)
            file3 = st.file_uploader(label='Excel file containing Base and Project cases',key='3',  type=['xlsx'])
            if file3:
                df_incremental_mapping = pd.read_excel(file3)
                dict_incremental_mapping = {df_incremental_mapping.loc[i,'Base']:df_incremental_mapping.loc[i,'Project'] for i in df_incremental_mapping.index}
                
                
                data_dict_base = st.session_state['data_dict_base'] 
                data_dict_project = st.session_state['data_dict_project'] 
                
                cases_base_correct = []
                cases_project_correct = []
                
                cases_base = data_dict_base['Metadata']['Cases']
                cases_project = data_dict_project['Metadata']['Cases']
                
                
                # Check if cases in the incremental mapping is actually found in the data
                for mapping_case_base in dict_incremental_mapping:
                    mapping_case_project = dict_incremental_mapping[mapping_case_base]
                    
                    if mapping_case_base in cases_base and mapping_case_project in cases_project:
                        cases_base_correct.append(mapping_case_base)
                        cases_project_correct.append(mapping_case_project)
                    else:
                        _ = dict_incremental_mapping.pop(mapping_case_base, None)
                        st.write("Removing {mapping_case_base} : {mapping_case_project} from mapping dict")
                
                data_dict_base['Metadata']['Cases'] = cases_base_correct
                data_dict_project['Metadata']['Cases'] = cases_project_correct
                

                for ensemble,cases in zip(['data_dict_base', 'data_dict_project'],[cases_base_correct,cases_project_correct]):
                    
                    data_dict = st.session_state[ensemble]
                    
                    for prop in data_dict['Field']:
                        df_temp = data_dict['Field'][prop].copy()
                        df_temp = df_temp[cases]
                        data_dict['Field'][prop] = df_temp
                    
                    for well in data_dict['Wells']:
                        for prop in data_dict['Wells'][well]:
                            df_temp = data_dict['Wells'][well][prop].copy()
                            df_temp = df_temp[cases]
                            data_dict['Wells'][well][prop] = df_temp
                    
                    for region in data_dict['Regions']:
                        for prop in data_dict['Regions'][region]:
                            df_temp = data_dict['Regions'][region][prop].copy()
                            df_temp = df_temp[cases]
                            data_dict['Regions'][region][prop] = df_temp
                        
                    st.session_state[ensemble] = data_dict
        
                st.session_state['dict_incremental_mapping'] = dict_incremental_mapping
                
                index = [(base, dict_incremental_mapping[base])for base in dict_incremental_mapping]
                st.session_state['case_groups'] = pd.DataFrame(index = index, columns = ['All'])
                st.session_state['case_groups']['All'] = True

    with col4:        
        if 'dict_incremental_mapping' in st.session_state:
            
            st.markdown("<h2 style='font-size:28px;'>Well Groups</h2>", unsafe_allow_html=True)
            file4 = st.file_uploader(label='Excel file with  well groups',key='4',  type=['xlsx'])
            if file4:
                df_well_groups = pd.read_excel(file4,index_col=0)
                st.session_state['well_groups'] = df_well_groups
        
                
    with col5:        
        if 'dict_incremental_mapping' in st.session_state:
            
            st.markdown("<h2 style='font-size:28px;'>Region Groups</h2>", unsafe_allow_html=True)
            file5 = st.file_uploader(label='Excel file with region groups',key='5',  type=['xlsx'])
            if file5:
                df_region_groups = pd.read_excel(file5,index_col=0)
                st.session_state['region_groups'] = df_region_groups
        
                

    
    with col1_:
        
        if 'dict_incremental_mapping' in st.session_state:
            
            st.markdown("<h2 style='font-size:28px;'>Case mapping Quality Control</h2>", unsafe_allow_html=True)
            st.success("Cases OK")
            st.write(st.session_state['dict_incremental_mapping'])
        
    with col3_:

        
        if 'dict_incremental_mapping' in st.session_state:
            
            st.markdown("<h2 style='font-size:28px;'>Data quality control</h2>", unsafe_allow_html=True)
            c1,c2,c3,c4 =st.columns(4)
               
            data_dict_base = st.session_state['data_dict_base'] 
            data_dict_project = st.session_state['data_dict_project'] 
            
            ################################################################# Dates
            dates_base = data_dict_base['Metadata']['Dates']
            dates_project = data_dict_project['Metadata']['Dates']
            
            with c1:
                if dates_base == dates_project:
                    st.success(f"Dates OK")
                else:
                    st.error(f"Dates are not equal in the two cases")
                
            ################################################################# Wells
            wells_base = data_dict_base['Metadata']['Wells']
            wells_project = data_dict_project['Metadata']['Wells']
            
            with c2:
                if wells_base == wells_project:
                    st.success(f"Wells OK")
                else:
                    st.error(f"Wells are not equal in the two cases")
            
            ################################################################# Regions
            regions_base = data_dict_base['Metadata']['Regions']
            regions_project = data_dict_project['Metadata']['Regions']
            
            with c3:
                if regions_base == regions_project:
                    st.success(f"Regions OK")
                else:
                    st.error(f"Regions are not equal in the two cases")
                    
            ################################################################# Properties
            properties_base = data_dict_base['Metadata']['Properties']
            properties_project = data_dict_project['Metadata']['Properties']
            
            with c4:
                if properties_base == properties_project:
                    st.success(f"Properties OK")
                else:
                    st.error(f"Properties are not equal in the two cases")
            
            
            df_display = pd.DataFrame({
                     'Dates': pd.Series(dates_base),
                     'Wells': pd.Series(wells_base),
                     'Regions': pd.Series(regions_base),
                     'Properties': pd.Series(properties_base),
                 })
           
            st.dataframe(df_display,use_container_width=True,height=1000)
            
                  
    with col5_:
        
        if 'dict_incremental_mapping' in st.session_state:
            if 'well_groups' in st.session_state:
                wells_base = list(data_dict_base['Metadata']['Wells'])
                well_groups_index = list(st.session_state['well_groups'].index)
            
                st.header("Well groups")
                
                if well_groups_index == wells_base:
                    st.success("Well Groups OK")
                else: 
                    st.error("Well Groups NOT OK")
                
                st.dataframe(st.session_state['well_groups'], use_container_width=True)
                
            if 'region_groups' in st.session_state:
                regions_base = [str(x) for x in data_dict_base['Metadata']['Regions']]
                regions_groups_index = [str(x) for x in (st.session_state['region_groups'].index)]
                
                st.header("Region groups")
                
                
                if regions_groups_index == regions_base:
                    st.success("Region Groups OK")
                else: 
                    st.error("Region Groups NOT OK")
                    
                st.dataframe(st.session_state['region_groups'], use_container_width=True)
        
        
 #######################################################Conversion Factors#########################################                  
                
            
if selected == "Sales Conversions":
        
    data_dict_base = st.session_state['data_dict_base']
    data_dict_project = st.session_state['data_dict_project']
    
    wells = data_dict_base['Metadata']['Wells']
    regions = data_dict_base['Metadata']['Regions']
    
    c1,c2,c3 =st.columns(3)
    
    with c1:
        st.header("Rate and fraction calculations:")
        
        if st.button("Rate and fraction calculations"):
            for ensemble in ['data_dict_base','data_dict_project']:
                
                data_dict = st.session_state[ensemble]
                
                cases = data_dict['Metadata']['Cases']
                
                dates = data_dict['Metadata']['Dates']
                days_in_period = [(dates[i+1]-dates[i]).days for i in range(len(dates[:-1]))]+[0]
                
                df_dates_in_period = pd.DataFrame(index = dates, columns = cases)
                for case in cases:
                    df_dates_in_period[case] = days_in_period
                
                
                for field in ['Field']:
                    
                    oil = data_dict['Field']['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    gas = data_dict['Field']['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0) 
                    water = data_dict['Field']['Cum water (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    
                    oil_vol = (oil.diff(periods=-1)*-1).fillna(0)
                    gas_vol = (gas.diff(periods=-1)*-1).fillna(0)
                    water_vol = (water.diff(periods=-1)*-1).fillna(0)
                    
                    oil_rate = oil_vol / df_dates_in_period
                    gas_rate = gas_vol / df_dates_in_period
                    water_rate = water_vol / df_dates_in_period
                    
                    liquid_rate = oil_rate + water_rate
                    wct = (water_rate / liquid_rate).fillna(0)
                    gor = (gas_rate / oil_rate).fillna(0)
                    
                    data_dict['Field']['Oil rate (sm3/d)'] = oil_rate
                    data_dict['Field']['Gas rate (sm3/d)'] = gas_rate
                    data_dict['Field']['Water rate (sm3/d)'] = water_rate
                    data_dict['Field']['Liquid rate (sm3/d)'] = liquid_rate
                    data_dict['Field']['Water cut (frac)'] = wct
                    data_dict['Field']['Gas oil ratio (sm3/sm3)'] = gor
                    
                for well in wells:
                    
                    oil = data_dict['Wells'][well]['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    gas = data_dict['Wells'][well]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0) 
                    water = data_dict['Wells'][well]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                                        
                    oil_vol = (oil.diff(periods=-1)*-1).fillna(0)
                    gas_vol = (gas.diff(periods=-1)*-1).fillna(0)
                    water_vol = (water.diff(periods=-1)*-1).fillna(0)
                    
                    oil_rate = oil_vol / df_dates_in_period
                    gas_rate = gas_vol / df_dates_in_period
                    water_rate = water_vol / df_dates_in_period
                    
                    liquid_rate = oil_rate + water_rate
                    wct = (water_rate / liquid_rate).fillna(0)
                    gor = (gas_rate / oil_rate).fillna(0)
                    
                    data_dict['Wells'][well]['Oil rate (sm3/d)'] = oil_rate
                    data_dict['Wells'][well]['Gas rate (sm3/d)'] = gas_rate
                    data_dict['Wells'][well]['Water rate (sm3/d)'] = water_rate
                    data_dict['Wells'][well]['Liquid rate (sm3/d)'] = liquid_rate
                    data_dict['Wells'][well]['Water cut (frac)'] = wct
                    data_dict['Wells'][well]['Gas oil ratio (sm3/sm3)'] = gor
                    
                for region in regions:
                    
                    oil = data_dict['Regions'][region]['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    gas = data_dict['Regions'][region]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    water = data_dict['Regions'][region]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                
                    oil_vol = (oil.diff(periods=-1)*-1).fillna(0)
                    gas_vol = (gas.diff(periods=-1)*-1).fillna(0)
                    water_vol = (water.diff(periods=-1)*-1).fillna(0)
                    
                    oil_rate = oil_vol / df_dates_in_period
                    gas_rate = gas_vol / df_dates_in_period
                    water_rate = water_vol / df_dates_in_period
                    
                    liquid_rate = oil_rate + water_rate
                    wct = (water_rate / liquid_rate).fillna(0)
                    gor = (gas_rate / oil_rate).fillna(0)
                    
                    data_dict['Regions'][region]['Oil rate (sm3/d)'] = oil_rate
                    data_dict['Regions'][region]['Gas rate (sm3/d)'] = gas_rate
                    data_dict['Regions'][region]['Water rate (sm3/d)'] = water_rate
                    data_dict['Regions'][region]['Liquid rate (sm3/d)'] = liquid_rate
                    data_dict['Regions'][region]['Water cut (frac)'] = wct
                    data_dict['Regions'][region]['Gas oil ratio (sm3/sm3)'] = gor
                
                for prop in ['Oil rate (sm3/d)', 'Water rate (sm3/d)', 'Gas rate (sm3/d)','Liquid rate (sm3/d)', 'Water cut (frac)','Gas oil ratio (sm3/sm3)']:
                    if prop not in data_dict['Metadata']['Properties']:
                        data_dict['Metadata']['Properties'].append(prop)
                
                st.session_state[ensemble] = data_dict 
        
            for prop in ['Oil rate (sm3/d)', 'Water rate (sm3/d)', 'Gas rate (sm3/d)','Liquid rate (sm3/d)', 'Water cut (frac)','Gas oil ratio (sm3/sm3)']:
                st.success(f'{prop} calculated and added to data')
            
    
    with c2:
        st.header("Standard conversions:")
        oil_per_gas = st.number_input("sm3 oil equivalent per sm3 gas",value=0.001,format="%.3f")
        stb_per_sm3 = st.number_input("stb per sm3",value = 6.293, format="%.4f")
        scf_per_sm3 = st.number_input("scf per sm3",value = 35.315, format="%.4f")
        
        if st.button("Calculate cum oil equivalents (sm3 o.e)"):
            for ensemble in ['data_dict_base','data_dict_project']:
                
                data_dict = st.session_state[ensemble]
                
                for field in ['Field']:
                    
                    oil_sm3 = data_dict['Field']['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    gas_sm3 = data_dict['Field']['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    
                    oe_sm3 = oil_sm3 + gas_sm3 *oil_per_gas
                    data_dict['Field']['Cum oil equivalent (sm3)'] = oe_sm3
                    
                for well in wells:
                    
                    oil_sm3 = data_dict['Wells'][well]['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    gas_sm3 = data_dict['Wells'][well]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    
                    oe_sm3 = oil_sm3 + gas_sm3 *oil_per_gas
                    
                    data_dict['Wells'][well]['Cum oil equivalent (sm3)'] = oe_sm3
                    
                    
                for region in regions:
                    
                    oil_sm3 = data_dict['Regions'][region]['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    gas_sm3 = data_dict['Regions'][region]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                    
                    oe_sm3 = oil_sm3 + gas_sm3 *oil_per_gas
                    data_dict['Regions'][region]['Cum oil equivalent (sm3)'] = oe_sm3
                            
                for prop in ['Cum oil equivalent (sm3)']:
                    if prop not in data_dict['Metadata']['Properties']:
                        data_dict['Metadata']['Properties'].append(prop)
                
                st.session_state[ensemble] = data_dict 
        
            for prop in ['Cum oil equivalent (sm3)']:
                st.success(f'{prop} calculated and added to data')
                

        
          
    with c3:
        st.header("Sales volume conversions:")
        
        with st.expander('Alvheim'):
            st.write('')
        with st.expander('Skarv'):
            st.write('')
        with st.expander('Sverdrup'):
            st.write('Sales gas = Produced gas - Injected gas')
            if st.button("Calculate:"):
                for ensemble in ['data_dict_base','data_dict_project']:
                    
                    data_dict = st.session_state[ensemble]
                    
                    for field in ['Field']:
                        
                        gas_prod = data_dict['Field']['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        gas_inj = data_dict['Field']['Cum gas injection (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        
                        gas_sales = gas_prod - gas_inj
                        data_dict['Field']['Cum sales gas (sm3)'] = gas_sales
                        
                    for well in wells:
                        
                        gas_prod = data_dict['Wells'][well]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        gas_inj = data_dict['Wells'][well]['Cum gas injection (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        
                        gas_sales = gas_prod - gas_inj
                        data_dict['Field']['Cum sales gas (sm3)'] = gas_sales
                        
                    for region in regions:
                        
                        gas_prod = data_dict['Regions'][region]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        gas_inj = data_dict['Regions'][region]['Cum gas injection (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        
                        gas_sales = gas_prod - gas_inj
                        data_dict['Field']['Cum sales gas (sm3)'] = gas_sales
                                
                    for prop in ['Cum sales gas (sm3)']:
                        if prop not in data_dict['Metadata']['Properties']:
                            data_dict['Metadata']['Properties'].append(prop)
                    
                    st.session_state[ensemble] = data_dict 
            
                for prop in ['Cum sales gas (sm3)']:
                    st.success(f'{prop} calculated and added to data')
            
        with st.expander('Grieg / Aasen'):
            st.write('')
        with st.expander('Yggdrasil'):
            st.write('')
        with st.expander('Vallhall'):
            oil_yield_from_gas = st.number_input("Oil Yield from Gas (stb oil / mscf gas)",value = 0.022, format="%.3f")
            gas_shrinkage_to_oil = st.number_input("Gas shrinkage to Oil (mscf gas / stb oil)",value = 0.006, format="%.3f")
            gas_shrinkage_to_ngl = st.number_input("Gas Shrinkage to NGL (scf gas / scf gas)",value = 0.8793, format="%.4f")
            ngl_yield_form_sales_gas = st.number_input("NGL yield from Sales Gas (stb ngl / mscf gas)",value = 0.046, format="%.4f")        
            
            if st.button("Calculate Valhall sales conversions"):
                for ensemble in ['data_dict_base','data_dict_project']:
                    
                    data_dict = st.session_state[ensemble]
                    
                    for field in ['Field']:
                        
                        oil_sm3 = data_dict['Field']['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        gas_sm3 = data_dict['Field']['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        
                        oil_mstb = oil_sm3 * stb_per_sm3 / 1000
                        gas_mmscf = gas_sm3 * scf_per_sm3 / 1e6
                        
                        oil_sales_mstb = oil_mstb + gas_mmscf * oil_yield_from_gas        
                        gas_sales_mmscf = gas_mmscf*gas_shrinkage_to_ngl - oil_sales_mstb*gas_shrinkage_to_oil
                        ngl_sales_mstb = gas_sales_mmscf*ngl_yield_form_sales_gas
                        oe_sales_mboe = oil_sales_mstb + gas_sales_mmscf /scf_per_sm3 *stb_per_sm3 *oil_per_gas + ngl_sales_mstb
                        
                        data_dict['Field']['Cum sales oil (mstb)'] = oil_sales_mstb
                        data_dict['Field']['Cum sales gas (mmscf)'] = gas_sales_mmscf
                        data_dict['Field']['Cum sales NGL(mstb)'] = ngl_sales_mstb
                        data_dict['Field']['Cum sales O.E. (mboe)'] = oe_sales_mboe
                        
                    for well in wells:
                        
                        oil_sm3 = data_dict['Wells'][well]['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        gas_sm3 = data_dict['Wells'][well]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        
                        oil_mstb = oil_sm3 * stb_per_sm3 / 1000
                        gas_mmscf = gas_sm3 * scf_per_sm3 / 1e6
                        
                        oil_sales_mstb = oil_mstb + gas_mmscf * oil_yield_from_gas        
                        gas_sales_mmscf = gas_mmscf*gas_shrinkage_to_ngl - oil_sales_mstb*gas_shrinkage_to_oil
                        ngl_sales_mstb = gas_sales_mmscf*ngl_yield_form_sales_gas
                        oe_sales_mboe = oil_sales_mstb + gas_sales_mmscf /scf_per_sm3 *stb_per_sm3 *oil_per_gas + ngl_sales_mstb
                        
                        data_dict['Wells'][well]['Cum sales oil (mstb)'] = oil_sales_mstb
                        data_dict['Wells'][well]['Cum sales gas (mmscf)'] = gas_sales_mmscf
                        data_dict['Wells'][well]['Cum sales NGL(mstb)'] = ngl_sales_mstb
                        data_dict['Wells'][well]['Cum sales O.E. (mboe)'] = oe_sales_mboe
                        
                    for region in regions:
                        
                        oil_sm3 = data_dict['Regions'][region]['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        gas_sm3 = data_dict['Regions'][region]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce').fillna(0)  
                        
                        oil_mstb = oil_sm3 * stb_per_sm3 / 1000
                        gas_mmscf = gas_sm3 * scf_per_sm3 / 1e6
                        
                        oil_sales_mstb = oil_mstb + gas_mmscf * oil_yield_from_gas        
                        gas_sales_mmscf = gas_mmscf*gas_shrinkage_to_ngl - oil_sales_mstb*gas_shrinkage_to_oil
                        ngl_sales_mstb = gas_sales_mmscf*ngl_yield_form_sales_gas
                        oe_sales_mboe = oil_sales_mstb + gas_sales_mmscf /scf_per_sm3 *stb_per_sm3 *oil_per_gas + ngl_sales_mstb
                        
                        data_dict['Regions'][region]['Cum sales oil (mstb)'] = oil_sales_mstb
                        data_dict['Regions'][region]['Cum sales gas (mmscf)'] = gas_sales_mmscf
                        data_dict['Regions'][region]['Cum sales NGL(mstb)'] = ngl_sales_mstb
                        data_dict['Regions'][region]['Cum sales O.E. (mboe)'] = oe_sales_mboe
                                
                    for prop in ['Cum sales oil (mstb)','Cum sales gas (mmscf)','Cum sales NGL(mstb)','Cum sales O.E. (mboe)']:
                        if prop not in data_dict['Metadata']['Properties']:
                            data_dict['Metadata']['Properties'].append(prop)
                    
                    st.session_state[ensemble] = data_dict 
            
                for prop in ['Cum sales oil (mstb)','Cum sales gas (mmscf)','Cum sales NGL(mstb)','Cum sales O.E. (mboe)']:
                    st.success(f'{prop} calculated and added to data')
            
    

########################################################################ANALYSIS#####################################################################           
      
elif selected == "Analysis":

        ########### setup widgets
        data_dict_base = st.session_state['data_dict_base']
        data_dict_project = st.session_state['data_dict_project']
        dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 

        # Ensure filtered_cases exists
        if 'filtered_cases' not in st.session_state:
            base_cases = data_dict_base['Metadata']['Cases']
            project_cases = data_dict_project['Metadata']['Cases']
            st.session_state['filtered_cases'] = list(zip(base_cases, project_cases))

        filtered_cases = st.session_state['filtered_cases']

        regions = sorted(data_dict_base['Metadata']['Regions'], key=sort_key)
        props = data_dict_base['Metadata']['Properties']
        dates = data_dict_base['Metadata']['Dates']
        wells = data_dict_base['Metadata']['Wells']
        well_groups = st.session_state['well_groups']
        region_groups = st.session_state['region_groups']
        cases_base = data_dict_base['Metadata']['Cases']
        cases_project = data_dict_project['Metadata']['Cases']

        tab1, tab2 = st.tabs(['Plots','Data'])
        with tab1:
            c1, c2, c3 = st.columns([5,1,30])
            with c1:
                with st.expander("Plot settings"):
                    plot_height = st.number_input("Plot height", 4, 16, 10)
                    base_tab, incr_tab = st.tabs(["Base / Project", "Incremental"])
                    with base_tab:
                        plot_base = st.checkbox('Plot base', True)
                        plot_project = st.checkbox('Plot project', True)
                    with incr_tab:
                        plot_incremental = st.checkbox('Plot incremental Histogram', True)
                        plot_scurve = st.checkbox('Plot incremental S-Curve', True)

                select_category = st.selectbox("Select Category", options=['Field', 'Region', 'Well', 'Well Group', 'Region Group'])
                selected_property = st.selectbox('Select property', props)

                if select_category == 'Field':
                    selected_identifier = 'Field'
                    df_base = data_dict_base['Field'][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_project = data_dict_project['Field'][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)

                elif select_category == 'Region':
                    selected_identifier = st.selectbox('Select region', regions)
                    df_base = data_dict_base['Regions'][selected_identifier][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_project = data_dict_project['Regions'][selected_identifier][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)

                elif select_category == 'Well':
                    selected_identifier = st.selectbox('Select well', wells)
                    df_base = data_dict_base['Wells'][selected_identifier][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_project = data_dict_project['Wells'][selected_identifier][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)

                elif select_category == 'Well Group':
                    selected_identifier = st.selectbox('Select Well Group', well_groups.columns)
                    dfs_base = [data_dict_base['Wells'][well][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                                for well in well_groups.index[well_groups[selected_identifier]==1]]
                    dfs_proj = [data_dict_project['Wells'][well][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                                for well in well_groups.index[well_groups[selected_identifier]==1]]
                    df_base = reduce(operator.add, dfs_base) if dfs_base else pd.DataFrame(index=dates, columns=cases_base).fillna(0)
                    df_project = reduce(operator.add, dfs_proj) if dfs_proj else pd.DataFrame(index=dates, columns=cases_project).fillna(0)

                elif select_category == 'Region Group':
                    selected_identifier = st.selectbox('Select Region Group', region_groups.columns)
                    dfs_base = [data_dict_base['Regions'][str(region)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                                for region in region_groups.index[region_groups[selected_identifier]==1]]
                    dfs_proj = [data_dict_project['Regions'][str(region)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                                for region in region_groups.index[region_groups[selected_identifier]==1]]
                    df_base = reduce(operator.add, dfs_base) if dfs_base else pd.DataFrame(index=dates, columns=cases_base).fillna(0)
                    df_project = reduce(operator.add, dfs_proj) if dfs_proj else pd.DataFrame(index=dates, columns=cases_project).fillna(0)

                df_base = df_base[ [x[0] for x in filtered_cases] ].fillna(0)
                df_project = df_project[ [x[1] for x in filtered_cases] ].fillna(0)

                df_incremental = pd.DataFrame(index=dates)
                for col_base in df_base.columns:
                    if col_base in dict_incremental_mapping:
                        col_project = dict_incremental_mapping[col_base]
                        df_incremental[col_project + ' - ' + col_base] = df_project[col_project] - df_base[col_base]

                selected_date = st.select_slider('Select date', options=dates, format_func=lambda date: date.strftime("%Y-%m-%d"))
                base_slice = df_base.loc[selected_date].fillna(0)
                project_slice = df_project.loc[selected_date].fillna(0)
                incremental_slice = df_incremental.loc[selected_date].fillna(0)
                incremental_slice_sorted = incremental_slice.sort_values()

                df_incremental_slice_cumprob = pd.DataFrame(index=incremental_slice_sorted.index)
                df_incremental_slice_cumprob['value'] = incremental_slice_sorted
                df_incremental_slice_cumprob['cum_prob'] = np.arange(len(incremental_slice_sorted), 0, -1) / len(incremental_slice_sorted)

            with c3:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        "Base/Project Time Series", 
                        "Base/Project Histogram",
                        "Incremental Time Series", 
                        "Incremental Histogram + S-Curve"
                    ],
                    specs=[[{}, {}], [{}, {"secondary_y": True}]],
                    vertical_spacing=0.15
                )

                # Define histogram bins
                if not base_slice.empty and not project_slice.empty:
                    bins = np.histogram_bin_edges(np.concatenate([base_slice, project_slice]), bins=30)
                else:
                    bins = np.linspace(0, 1, 31)  # default fallback

                # --- Top left: Base / Project Time Series
                if plot_base:
                    for col in df_base.columns:
                        fig.add_trace(go.Scatter(
                            x=df_base.index, y=df_base[col], mode='lines',
                            name=f"Base: {col}", line=dict(color='grey', width=1), opacity=0.5
                        ), row=1, col=1)

                if plot_project:
                    for col in df_project.columns:
                        fig.add_trace(go.Scatter(
                            x=df_project.index, y=df_project[col], mode='lines',
                            name=f"Project: {col}", line=dict(color='blue', width=1), opacity=0.5
                        ), row=1, col=1)

                fig.add_vline(x=selected_date, line_dash='dash', line_color='black', row=1, col=1)

                # --- Top right: Base / Project Histogram
                if plot_base:
                    fig.add_trace(go.Histogram(
                        x=base_slice, name="Base", marker_color='grey', opacity=0.75,
                        xbins=dict(start=bins[0], end=bins[-1], size=(bins[1] - bins[0])),
                        marker_line=dict(width=1.5, color='black')
                    ), row=1, col=2)

                if plot_project:
                    fig.add_trace(go.Histogram(
                        x=project_slice, name="Project", marker_color='blue', opacity=0.5,
                        xbins=dict(start=bins[0], end=bins[-1], size=(bins[1] - bins[0])),
                        marker_line=dict(width=1.5, color='black')
                    ), row=1, col=2)

                # --- Bottom left: Incremental Time Series
                if plot_incremental:
                    for col in df_incremental.columns:
                        fig.add_trace(go.Scatter(
                            x=df_incremental.index, y=df_incremental[col], mode='lines',
                            name=f"Incremental: {col}", line=dict(color='red'), opacity=0.5
                        ), row=2, col=1)

                    fig.add_vline(x=selected_date, line_dash='dash', line_color='black', row=2, col=1)

                # --- Bottom right: Incremental Histogram + S-Curve
                if plot_incremental:
                    fig.add_trace(go.Histogram(
                        x=incremental_slice, name="Incremental", marker_color='red', opacity=0.5
                    ), row=2, col=2, secondary_y=False)

                if plot_scurve:
                    fig.add_trace(go.Scatter(
                        x=df_incremental_slice_cumprob['value'],
                        y=df_incremental_slice_cumprob['cum_prob'],
                        mode='markers', name="Cumulative Probability",
                        marker=dict(color='red', line=dict(color='black', width=1))
                    ), row=2, col=2, secondary_y=True)

                    # P10, P50, P90 lines
                    for q, color in zip([0.1, 0.5, 0.9], ['firebrick', 'blue', 'green']):
                        val = incremental_slice.quantile(q)
                        fig.add_vline(x=val, line_dash="dashdot", line_color=color, row=2, col=2)
                        fig.add_hline(y=q, line_dash="dashdot", line_color=color, row=2, col=2, secondary_y=True)

                # --- Axis and Layout
                fig.update_xaxes(title_text="Time (days)", row=1, col=1)
                fig.update_xaxes(title_text=f"{selected_property} @ {selected_date.strftime('%Y-%m-%d')}", row=1, col=2)
                fig.update_xaxes(title_text="Time (days)", row=2, col=1)
                fig.update_xaxes(title_text=f"{selected_property} @ {selected_date.strftime('%Y-%m-%d')}", row=2, col=2)

                fig.update_yaxes(title_text=selected_property, row=1, col=1)
                fig.update_yaxes(title_text="Count", row=1, col=2)
                fig.update_yaxes(title_text=selected_property, row=2, col=1)
                fig.update_yaxes(title_text="Count", row=2, col=2)
                fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2, secondary_y=True)

                fig.update_layout(
                    height=plot_height * 100,
                    title=dict(
                        text=f"{selected_identifier}: {selected_property}",
                        font=dict(size=22),
                        x=0.0,
                        xanchor='left'
                    ),
                    font=dict(size=16),
                    showlegend=True,
                    template="plotly_white",
                    barmode="overlay"
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(df_base, use_container_width=True)
            st.dataframe(df_project, use_container_width=True)
            st.dataframe(df_incremental, use_container_width=True)


        
#############################################################################CROSSPLOT#####################################################################################       
elif selected == "Crossplot":

    ########### setup widgets
    
    data_dict_base = st.session_state['data_dict_base']
    data_dict_project = st.session_state['data_dict_project']
    
    regions = data_dict_base['Metadata']['Regions']
    regions = sorted(regions, key=sort_key)
    props = data_dict_base['Metadata']['Properties']
    dates = data_dict_base['Metadata']['Dates']
    wells = data_dict_base['Metadata']['Wells']
    dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 
    
    
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
        c1,c2,c3 = st.columns([5,1,30])
        
        with c1:
            
            with st.expander("Plot settings"):
                plot_height = st.number_input("Plot height",4,16,10)
                tab1,tab2=st.tabs(["Base / Project","Incremental"])
                with tab1:
                    plot_base = st.checkbox('Plot base',True)
                    plot_project = st.checkbox('Plot project',True)
                    
                    
                with tab2:
                    plot_incremental = st.checkbox('Plot incremental Histogram',True)
                    plot_scurve = st.checkbox('Plot incremental S-Curve',True)
                    override_axis_incremental = st.checkbox("Override Incremental Axes",)
                    if override_axis_incremental:
                        col1,col2 = st.columns(2)
                        with col1:
                            incremental_yaxis_min = st.number_input("Incremental Axis Min")
                        with col2:
                            incremental_yaxis_max = st.number_input("Incremental Axis Max")
            
            with st.expander("X-Axis variable:",expanded=True):
                
                x_selected_category = st.selectbox('Select category (X)', options = ['Field','Region','Well'])
                x_selected_type = st.selectbox('Select source (X)',['Base','Project','Incremental'])
                x_selected_property = st.selectbox('Select property (X)', props)
        
                if x_selected_category == 'Field':
                    x_selected_identifier = 'Field'
                    x_df_base = data_dict_base['Field'][x_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    x_df_project = data_dict_project['Field'][x_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    
                if x_selected_category == 'Region':
                    x_selected_identifier = st.selectbox('Select region (X)', regions)
                    x_df_base = data_dict_base['Regions'][x_selected_identifier][x_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    x_df_project = data_dict_project['Regions'][x_selected_identifier][x_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    
                if x_selected_category == 'Well':
                    x_selected_identifier = st.selectbox('Select well (X)', wells)
                    x_df_base = data_dict_base['Wells'][x_selected_identifier][x_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    x_df_project = data_dict_project['Wells'][x_selected_identifier][x_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                
                #Calculate incremental
                x_df_incremental = pd.DataFrame(index=dates)
                for col_base in x_df_base.columns:
                    if col_base in dict_incremental_mapping:
                        col_project = dict_incremental_mapping[col_base]
                        x_df_incremental[col_project+' - '+col_base] = x_df_project[col_project]-x_df_base[col_base]
            
                #Extract timeslice
                x_selected_date = st.select_slider('Select date (X)',options = dates, format_func=lambda date: date.strftime("%Y-%m-%d"))
                x_base_slice = x_df_base.loc[x_selected_date]
                x_project_slice =  x_df_project.loc[x_selected_date]
                x_incremental_slice =  x_df_incremental.loc[x_selected_date]
                
                if x_selected_type == 'Base':
                    x_slice = x_base_slice
                if x_selected_type == 'Project':
                    x_slice = x_project_slice
                if x_selected_type == 'Incremental':
                    x_slice = x_incremental_slice
                    
            with st.expander("X-Axis variable:",expanded=True):
                
                y_selected_category = st.selectbox('Select category (Y)', options = ['Field','Region','Well'])
                y_selected_type = st.selectbox('Select source (Y)',['Base','Project','Incremental'])
                y_selected_property = st.selectbox('Select property (Y)', props)
        
                if y_selected_category == 'Field':
                    y_selected_identifier = 'Field'
                    y_df_base = data_dict_base['Field'][y_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    y_df_project = data_dict_project['Field'][y_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    
                if y_selected_category == 'Region':
                    y_selected_identifier = st.selectbox('Select region (Y)', regions)
                    y_df_base = data_dict_base['Regions'][y_selected_identifier][y_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    y_df_project = data_dict_project['Regions'][y_selected_identifier][y_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    
                if y_selected_category == 'Well':
                    y_selected_identifier = st.selectbox('Select well (Y)', wells)
                    y_df_base = data_dict_base['Wells'][y_selected_identifier][y_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                    y_df_project = data_dict_project['Wells'][y_selected_identifier][y_selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)   
                
                #Calculate incremental
                y_df_incremental = pd.DataFrame(index=dates)
                for col_base in y_df_base.columns:
                    
                    if col_base in dict_incremental_mapping:
                        col_project = dict_incremental_mapping[col_base]
                        y_df_incremental[col_project+' - '+col_base] = y_df_project[col_project]-y_df_base[col_base]
            
                #Extract timeslice
                y_selected_date = st.select_slider('Select date (Y)',options = dates, format_func=lambda date: date.strftime("%Y-%m-%d"))
                y_base_slice = y_df_base.loc[y_selected_date]
                y_project_slice =  y_df_project.loc[y_selected_date]
                y_incremental_slice =  y_df_incremental.loc[y_selected_date]
                
                if y_selected_type == 'Base':
                    y_slice = y_base_slice
                if y_selected_type == 'Project':
                    y_slice = y_project_slice
                if y_selected_type == 'Incremental':
                    y_slice = y_incremental_slice
        
            
            
        ########### plot data
        with c3:
            fig,ax = plt.subplots(figsize=(18,plot_height))
            
            
            ax.scatter(x_slice,y_slice,color = 'grey',alpha=0.5, edgecolor = 'black')
                 
            
            ax.axvline(x_slice.quantile(0.9), color='green', linestyle='dashdot')
            ax.axvline(x_slice.quantile(0.5), color='blue', linestyle='dashdot')
            ax.axvline(x_slice.quantile(0.1), color='firebrick', linestyle='dashdot')
            
            ax.axhline(y_slice.quantile(0.9), color='green', linestyle='dashdot')
            ax.axhline(y_slice.quantile(0.5), color='blue', linestyle='dashdot')
            ax.axhline(y_slice.quantile(0.1), color='firebrick', linestyle='dashdot')
        
            ax.set_ylabel(f'{y_selected_type}: {y_selected_property} @ {y_selected_date.strftime("%Y-%m-%d")}')
            ax.set_xlabel(f'{x_selected_type}: {x_selected_property} @ {x_selected_date.strftime("%Y-%m-%d")}')
                
            ax.grid()
            ax.legend()
          
            st.pyplot(fig,use_container_width=True)           
            
    with tab2:
        
        st.write(f'X-Axis: {x_selected_type}: {x_selected_property} @ {x_selected_date.strftime("%Y-%m-%d")}')
        st.dataframe(x_slice,use_container_width=True)
        st.write(f'Y-Axis: {y_selected_type}: {y_selected_property} @ {y_selected_date.strftime("%Y-%m-%d")}')
        st.dataframe(y_slice,use_container_width=True)

######################################################################BOXPLOTS#####################################################################
elif selected == "Boxplots":
    if 'data_dict_base' not in st.session_state or 'data_dict_project' not in st.session_state:
        st.error("Missing data. Please go to the Upload page and load both base and project datasets.")
        st.stop()
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
        c1,c2,c3 =st.columns([5,1,30])
    ########### setup widgets
        with c1:
    
            data_dict_base = st.session_state['data_dict_base']
            data_dict_project = st.session_state['data_dict_project']
            
            regions = data_dict_base['Metadata']['Regions']
            regions = sorted(regions, key=sort_key)
            props = data_dict_base['Metadata']['Properties']
            dates = data_dict_base['Metadata']['Dates']
            wells = data_dict_base['Metadata']['Wells']
            region_groups = st.session_state['region_groups']
            well_groups = st.session_state['well_groups']
                                                 
                                              
            dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 
            
            with st.expander("Plot settings"):
                plot_height = st.number_input("Plot height",4,16,10)
    
            selected_category = st.selectbox("Select category",options=["Regions","Wells", "Well Groups", "Region Groups"])
            selected_plot_type = st.selectbox('Select ensemble', ["Base","Project","Incremental"])
            selected_property = st.selectbox('Select property', props)
            selected_date = st.select_slider('Select date',options = dates, format_func=lambda date: date.strftime("%Y-%m-%d"))
        
            box_data_base = pd.DataFrame()
            box_data_project = pd.DataFrame()
            box_data_incremental = pd.DataFrame()
        
            if selected_category == 'Regions':
                selected_identifiers = st.multiselect("Selected regions", regions, regions)
                for region in selected_identifiers:
                    df_base = data_dict_base['Regions'][region][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_project = data_dict_project['Regions'][region][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                                
                    df_base = df_base.fillna(0)
                    df_project = df_project.fillna(0)
                    
                    
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
                    
                    
                    
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        if col_base in dict_incremental_mapping:
                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
                    
                    base_slice = df_base.loc[selected_date]
                    project_slice =  df_project.loc[selected_date]
                    incremental_slice =  df_incremental.loc[selected_date]
                    
                    
                    box_data_base[region] = base_slice
                    box_data_project[region] = project_slice
                    box_data_incremental[region] = incremental_slice
                    
            if selected_category == 'Wells':
                
                if st.checkbox("Filter on well group"):
                    
                    well_group = st.selectbox(f"Select Well Group",options = well_groups.columns)
                    selectable_wells = well_groups.index[well_groups[well_group]==1]
                else:
                    selectable_wells = wells
                
                selected_identifiers = st.multiselect("Selected wells", selectable_wells, selectable_wells)
                for region in selected_identifiers:
                    df_base = data_dict_base['Wells'][region][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_project = data_dict_project['Wells'][region][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                                
                    df_base = df_base.fillna(0)
                    df_project = df_project.fillna(0)
                    
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
                    
                    
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        if col_base in dict_incremental_mapping:    
                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
                        
                    base_slice = df_base.loc[selected_date]
                    project_slice =  df_project.loc[selected_date]
                    incremental_slice =  df_incremental.loc[selected_date]
                    
                    
                    box_data_base[region] = base_slice
                    box_data_project[region] = project_slice
                    box_data_incremental[region] = incremental_slice
    
            if selected_category == 'Region Groups':
                selected_identifiers = st.multiselect("Select Region Group", options = region_groups.columns,default = region_groups.columns)
                    
                for region_group in selected_identifiers:
                
                    dfs_temp=[]
                    for region in region_groups.index[region_groups[region_group]==1]:
                        
                        df_temp = data_dict_base['Regions'][str(region)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                        dfs_temp.append(df_temp)
                    if len(dfs_temp)>0:
                        df_base = reduce(operator.add, dfs_temp)
                    else:
                        df_base = pd.DataFrame(index=dates,columns=cases_base).fillna(0)
                        
                    dfs_temp=[]
                    for region in region_groups.index[region_groups[region_group]==1]:
                        df_temp = data_dict_project['Regions'][str(region)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                        dfs_temp.append(df_temp)
                    if len(dfs_temp)>0:
                        df_project = reduce(operator.add, dfs_temp)
                    else:
                        df_project = pd.DataFrame(index=dates,columns=cases_project).fillna(0)
    
    
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
                    
    
    
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        if col_base in dict_incremental_mapping:    
                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
               
                    base_slice = df_base.loc[selected_date]
                    project_slice =  df_project.loc[selected_date]
                    incremental_slice =  df_incremental.loc[selected_date]
                    
                    box_data_base[region_group] = base_slice
                    box_data_project[region_group] = project_slice
                    box_data_incremental[region_group] = incremental_slice
                    
            if selected_category == 'Well Groups':
                selected_identifiers = st.multiselect("Select Well Groups", options = well_groups.columns,default = well_groups.columns)
                    
                for well_group in selected_identifiers:
                    dfs_temp=[]
                    for well in well_groups.index[well_groups[well_group]==1]:
                        
                        df_temp = data_dict_base['Wells'][str(well)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                        dfs_temp.append(df_temp)
                    if len(dfs_temp)>0:
                        df_base = reduce(operator.add, dfs_temp)
                    else:
                        df_base = pd.DataFrame(index=dates,columns=cases_base).fillna(0)
                        
                    dfs_temp=[]
                    for well in well_groups.index[well_groups[well_group]==1]:
                        df_temp = data_dict_project['Wells'][str(well)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                        dfs_temp.append(df_temp)
                    if len(dfs_temp)>0:
                        df_project = reduce(operator.add, dfs_temp)
                    else:
                        df_project = pd.DataFrame(index=dates,columns=cases_project).fillna(0)
            
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
            
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        if col_base in dict_incremental_mapping:    
                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
               
                    base_slice = df_base.loc[selected_date]
                    project_slice =  df_project.loc[selected_date]
                    incremental_slice =  df_incremental.loc[selected_date]
                    
                    box_data_base[well_group] = base_slice
                    box_data_project[well_group] = project_slice
                    box_data_incremental[well_group] = incremental_slice
    
    
            if selected_plot_type == 'Base':
                df = box_data_base.copy()
            elif selected_plot_type == 'Project':
                df = box_data_project.copy()
            elif selected_plot_type == 'Incremental':
                df = box_data_incremental.copy()
        
        
        with c3:
            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Box(
                #y=data,
                name='Region A',
                marker_color='blue',          # Box fill color
                line=dict(color='black', width=2), # Outline color
                fillcolor='blue',             # (Optional) explicitly set box fill color
                boxpoints='all',                   # Optional: 'all', 'outliers', or False
                jitter=0.5                         # Spread out the points
           ))

            for i, col in enumerate(df.columns):
                col_data = df[col]
                min_val = col_data.min()
                p10 = col_data.quantile(0.9)
                p50 = col_data.quantile(0.5)
                p90 = col_data.quantile(0.1)
                max_val = col_data.max()

                # Bar for P10–P90 band
                fig.add_trace(go.Bar(
                    x=[col],
                    y=[p10 - p90],
                    base=[p90],
                    marker_color='lightblue',
                    marker_line=dict(color='black', width=1),
                    name="P10 - P90" if i == 0 else "",
                    showlegend=(i == 0)
                ))

                # P50 marker
                fig.add_trace(go.Scatter(
                    x=[col],
                    y=[p50],
                    mode='markers',
                    marker=dict(color='black', symbol='circle', size=5),
                    name="P50" if i == 0 else "",
                    showlegend=(i == 0)
                ))

                # Min to P90
                fig.add_trace(go.Scatter(
                    x=[col, col],
                    y=[min_val, p90],
                    mode='lines',
                    line=dict(color='black'),
                    name="Min" if i == 0 else "",
                    showlegend=(i == 0)
                ))

                # Min 
                fig.add_shape(
                    type="line",
                    x0=i - 0.1, x1=i + 0.1,    # assuming xaxis is type='linear' (category index)
                    y0=min_val, y1=min_val,
                    line=dict(color="black", width=2),
                    xref="x", yref="y"
                )


                # Max
                fig.add_shape(
                    type="line",
                    x0=i - 0.1, x1=i + 0.1,    # assuming xaxis is type='linear' (category index)
                    y0=max_val, y1=max_val,
                    line=dict(color="black", width=2),
                    xref="x", yref="y"
                )
                # P10 to Max
                fig.add_trace(go.Scatter(
                    x=[col, col],
                    y=[p10, max_val],
                    mode='lines',
                    line=dict(color='black'),
                    name="Max" if i == 0 else "",
                    showlegend=(i == 0)
                ))

            # Layout
            fig.update_layout(
                title=f"{selected_category}: {selected_property} @ {selected_date.strftime('%Y-%m-%d')}",
                yaxis_title=selected_property,
                xaxis_title=selected_category,
                template="plotly_white",
                height=1000,
                barmode='overlay',
                xaxis_tickangle=45
            )

            st.plotly_chart(fig, use_container_width=True)
        
    
    with tab2:            
        st.dataframe(box_data_base,use_container_width=True)           
        st.dataframe(box_data_project,use_container_width=True)           
        st.dataframe(box_data_incremental,use_container_width=True)           
        
###########################################################################Waterfall#####################################################################################
     
elif selected == "Waterfall":
    
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
        c1,c2,c3 = st.columns([5,1,30])
        with c1:
    ########### setup widgets
            
            data_dict_base = st.session_state['data_dict_base']
            data_dict_project = st.session_state['data_dict_project']
            
            regions = data_dict_base['Metadata']['Regions']
            regions = sorted(regions, key=sort_key)
            props = data_dict_base['Metadata']['Properties']
            dates = data_dict_base['Metadata']['Dates']
            wells = list(data_dict_base['Metadata']['Wells'])
            region_groups = st.session_state['region_groups']
            well_groups = st.session_state['well_groups']
                                                         
            dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 
    
            with st.expander("Plot settings"):
                plot_height = st.number_input("Plot height",4,16,10)
                merge_zeros = st.checkbox('Merge values', props, False)
                if merge_zeros:
                    cutoff = st.number_input("Cutoff value",value = 1)
                ylim_setting = st.number_input('Y axis scale',min_value = 0.001,max_value = 0.100,value = 0.050,step =0.001,format="%.3f")
    
    
            selected_category = st.selectbox("Select category",options=["Regions","Wells","Well Groups","Region Groups"])
            selected_property = st.selectbox('Select property', props)
            selected_date = st.select_slider('Select date',options = dates, format_func=lambda date: date.strftime("%Y-%m-%d"))
        
            box_data_base = pd.DataFrame()
            box_data_project = pd.DataFrame()
            box_data_incremental = pd.DataFrame()
            
            if selected_category == 'Regions':
                identifiers =  st.multiselect('Select regions',options=regions,default=regions)
                
                for region in identifiers:
                    df_base = data_dict_base['Regions'][region][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_project = data_dict_project['Regions'][region][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                            
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
                    
                    
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        
                        if col_base in dict_incremental_mapping:
                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
                    
                    base_slice = df_base.loc[selected_date]
                    project_slice =  df_project.loc[selected_date]
                    incremental_slice =  df_incremental.loc[selected_date]
                    
                    box_data_base[region] = base_slice
                    box_data_project[region] = project_slice
                    box_data_incremental[region] = incremental_slice
                    
            if selected_category == 'Wells':
                
                if st.checkbox("Filter on well group"):
                    
                    well_group = st.selectbox(f"Select Well Group",options = well_groups.columns)
                    selectable_wells = well_groups.index[well_groups[well_group]==1]
                else:
                    selectable_wells = wells
                
                identifiers = st.multiselect('Select wells',options=selectable_wells,default=selectable_wells)
                for well in identifiers:
                    df_base = data_dict_base['Wells'][well][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_project = data_dict_project['Wells'][well][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
            
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        
                        if col_base in dict_incremental_mapping:
                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
                    
                    base_slice = df_base.loc[selected_date]
                    project_slice =  df_project.loc[selected_date]
                    incremental_slice =  df_incremental.loc[selected_date]
                    
                    box_data_base[well] = base_slice
                    box_data_project[well] = project_slice
                    box_data_incremental[well] = incremental_slice
            
            if selected_category == 'Well Groups':
                identifiers = st.multiselect("Select well Group", options = well_groups.columns,default = well_groups.columns)
                    
                for well_group in identifiers:
                
                    dfs_temp=[]
                    for well in well_groups.index[well_groups[well_group]==1]:
                        
                        df_temp = data_dict_base['Wells'][str(well)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                        dfs_temp.append(df_temp)
                    if len(dfs_temp)>0:
                        df_base = reduce(operator.add, dfs_temp)
                    else:
                        df_base = pd.DataFrame(index=dates,columns=cases_base).fillna(0)
                        
                    dfs_temp=[]
                    for well in well_groups.index[well_groups[well_group]==1]:
                        df_temp = data_dict_project['Wells'][str(well)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                        dfs_temp.append(df_temp)
                    if len(dfs_temp)>0:
                        df_project = reduce(operator.add, dfs_temp)
                    else:
                        df_project = pd.DataFrame(index=dates,columns=cases_project).fillna(0)
    
    
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
                    
    
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        if col_base in dict_incremental_mapping:    
                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
               
                    base_slice = df_base.loc[selected_date]
                    project_slice =  df_project.loc[selected_date]
                    incremental_slice =  df_incremental.loc[selected_date]
                    
                    box_data_base[well_group] = base_slice
                    box_data_project[well_group] = project_slice
                    box_data_incremental[well_group] = incremental_slice
                
            if selected_category == 'Region Groups':
                identifiers = st.multiselect("Select Region Group", options = region_groups.columns,default = region_groups.columns)
                    
                for region_group in identifiers:
                
                    dfs_temp=[]
                    for region in region_groups.index[region_groups[region_group]==1]:
                        
                        df_temp = data_dict_base['Regions'][str(region)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                        dfs_temp.append(df_temp)
                    if len(dfs_temp)>0:
                        df_base = reduce(operator.add, dfs_temp)
                    else:
                        df_base = pd.DataFrame(index=dates,columns=cases_base).fillna(0)
                        
                    dfs_temp=[]
                    for region in region_groups.index[region_groups[region_group]==1]:
                        df_temp = data_dict_project['Regions'][str(region)][selected_property].apply(pd.to_numeric, errors='coerce').fillna(0)
                        dfs_temp.append(df_temp)
                    if len(dfs_temp)>0:
                        df_project = reduce(operator.add, dfs_temp)
                    else:
                        df_project = pd.DataFrame(index=dates,columns=cases_project).fillna(0)
    
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
            
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        if col_base in dict_incremental_mapping:    
                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
               
                    base_slice = df_base.loc[selected_date]
                    project_slice =  df_project.loc[selected_date]
                    incremental_slice =  df_incremental.loc[selected_date]
                    
                    box_data_base[region_group] = base_slice
                    box_data_project[region_group] = project_slice
                    box_data_incremental[region_group] = incremental_slice
            
            base = box_data_base.mean().sum()
            project = box_data_project.mean().sum()
            incremental = box_data_incremental.mean()
            
            ############################################################
            df_temp = pd.DataFrame(index = identifiers)
            df_temp['Value'] = list(incremental)
            
            if merge_zeros:
                df_main = df_temp[df_temp['Value'].abs()>=cutoff]
                df_other = df_temp[df_temp['Value'].abs()<cutoff]            
    
                df_waterfall = pd.DataFrame(index = ['Base'] + list(df_main.index) + ['Other']+['Project'],columns = ['Value'])
                df_waterfall['Value'] = [base] + list(df_main['Value'])+ [df_other['Value'].sum()]+[project]
    
            else:
                df_waterfall = pd.DataFrame(index = ['Base'] + list(df_temp.index) + ['Project'],columns = ['Value'])
                df_waterfall['Value'] = [base] + list(df_temp['Value']) +[project]
            
            
            df_waterfall['Category'] = df_waterfall.index
            
            for i, ind in enumerate(df_waterfall.index):
                if i == 0 or i == len(df_waterfall.index)-1:
                    df_waterfall.loc[ind,'Bottom']=0
                    continue
                df_waterfall.loc[df_waterfall.index[i],'Bottom'] = df_waterfall.loc[df_waterfall.index[i-1],'Value'] + df_waterfall.loc[df_waterfall.index[i-1],'Bottom']
            
            df_waterfall['Total'] = df_waterfall['Value']+df_waterfall['Bottom']
            df_waterfall["Color"] = ["blue"] + ["green" if v > 0 else "red" for v in df_waterfall["Value"][1:-1]] + ["blue"]

        
############################
   
    import plotly.graph_objects as go

    with c3:
        fig = go.Figure()

        # Add waterfall bars
        fig.add_trace(go.Bar(
            x=df_waterfall["Category"],
            y=df_waterfall["Value"],
            base=df_waterfall["Bottom"],
            marker_color=df_waterfall["Color"],
            marker_line=dict(color='black', width=1),
            text=[f"{v:.3e}" for v in df_waterfall["Value"]],
            textposition='outside',
            textfont=dict(size=16),
            name="Incremental Values"
        ))

        # Adjust layout
        fig.update_layout(
             
            yaxis=dict(
                title=f'{selected_property} @ {selected_date.strftime("%Y-%m-%d")}',
                range=[
                    df_waterfall["Total"].min() * (1 - ylim_setting),
                    df_waterfall["Total"][1:-1].max() * (1 + ylim_setting)
                ]
            ),
            xaxis=dict(
                title=selected_category,
                tickangle=30
            ),
            title="Waterfall chart for MEAN Incremental values",
            template="plotly_white",
            showlegend=False,
            height=plot_height * 100
        )

        st.plotly_chart(fig, use_container_width=True)

          
        
###########################################################################################################CASE SELECTION###########################################################################################################
elif selected == "Case selection":
    
    # Retrieve base data from session state
    data_dict_base = st.session_state['data_dict_base']
    data_dict_project = st.session_state['data_dict_project']
    
    regions = data_dict_base['Metadata']['Regions']
    regions = list(sorted(regions, key=sort_key))
    props = data_dict_base['Metadata']['Properties']
    dates = data_dict_base['Metadata']['Dates']
    wells = list(data_dict_base['Metadata']['Wells'])
    well_groups = st.session_state['well_groups']
    region_groups = st.session_state['region_groups']

                                         
    dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 
    
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
        c1,c2,c3 = st.columns([5,1,30])
        with c1:
            # Allow the user to select the number of expander groups
            #num_groups = st.number_input("Select number of variable sets (maximum 3)", min_value=1, max_value=3, value=1)
            
            with st.expander("Plot settings"):
                plot_height = st.number_input("Plot height",1,16,4)
            num_groups = st.number_input('Number of properties:',1,10,1)
            select_source = st.selectbox(f"Select Source",options = ['Base','Project','Incremental',])
        
            selected_identifiers = []
            selected_props = []
            selected_dates_objects = []
            selected_dates_strings = []
            weights = []
            
            p50_cases = pd.DataFrame()
            
            dfs = []
            dfs_cumprob = []
        
            # Create a 3-column, 2-row subplot grid (for now.., remember: num_groups is hardcoded and set to 1)
            fig, axs = plt.subplots(num_groups,2, figsize=(12, plot_height*num_groups))
            
            for i in range(num_groups):
                with st.expander(f"Property {i+1}", expanded=True):
                
                    select_category = st.selectbox(f"Select Category {i+1}",options = ['Field','Region','Well','Well Group','Region Group'])
                    selected_props.append(st.selectbox(f'Property {i+1}', props, index=min(i,len(props)-1)))
    
                    if select_category == 'Field':
                        selected_identifiers.append('Field')
                        selected_identifier = selected_identifiers[i]
                        
                        df_base = data_dict_base['Field'][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)                  
                        df_project = data_dict_project['Field'][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)          
                        
                    if select_category == 'Region':
                        selected_identifiers.append(st.selectbox(f'Select region {i+1}', regions))
                        selected_identifier = selected_identifiers[i]
                        
                        df_base = data_dict_base['Regions'][selected_identifier][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)                  
                        df_project = data_dict_project['Regions'][selected_identifier][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)          
                        
                    if select_category == 'Well':
                        selected_identifiers.append(st.selectbox(f'Select well {i+1}', wells))
                        selected_identifier = selected_identifiers[i]
                        
                        df_base = data_dict_base['Wells'][selected_identifier][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)       
                        df_project = data_dict_project['Wells'][selected_identifier][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)   
                        
                    if select_category == 'Well Group':
                        selected_identifiers.append(st.selectbox(f"Select Well Group {i+1}",options = well_groups.columns))
                        selected_identifier = selected_identifiers[i]

                        
                        dfs_temp=[]
                        for well in well_groups.index[well_groups[selected_identifier]==1]:
                            df_temp = data_dict_base['Wells'][well][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)
                            dfs_temp.append(df_temp)
                        if len(dfs_temp)>0:
                            df_base = reduce(operator.add, dfs_temp)
                        else:
                            df_base = pd.DataFrame(index=dates,columns=cases_base).fillna(0)
                            
                        dfs_temp=[]
                        for well in well_groups.index[well_groups[selected_identifier]==1]:
                            df_temp = data_dict_project['Wells'][well][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)
                            dfs_temp.append(df_temp)
                        if len(dfs_temp)>0:
                            df_project = reduce(operator.add, dfs_temp)
                        else:
                            df_project = pd.DataFrame(index=dates,columns=cases_project).fillna(0)
                                            
                    if select_category == 'Region Group':
                        selected_identifiers.append(st.selectbox(f"Select Region Group {i+1}",options = region_groups.columns))
                        selected_identifier = selected_identifiers[i]

                        
                        dfs_temp=[]
                        for region in region_groups.index[region_groups[selected_identifier]==1]:
                            region = str(region)
                            
                            df_temp = data_dict_base['Regions'][region][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)
                            dfs_temp.append(df_temp)
                        if len(dfs_temp)>0:
                            df_base = reduce(operator.add, dfs_temp)
                        else:
                            df_base = pd.DataFrame(index=dates,columns=cases_base).fillna(0)
                            
                        dfs_temp=[]
                        for region in region_groups.index[region_groups[selected_identifier]==1]:
                            region = str(region)
                            
                            df_temp = data_dict_project['Regions'][region][selected_props[i]].apply(pd.to_numeric, errors='coerce').fillna(0)
                            dfs_temp.append(df_temp)
                        if len(dfs_temp)>0:
                            df_project = reduce(operator.add, dfs_temp)
                        else:
                            df_project = pd.DataFrame(index=dates,columns=cases_project).fillna(0)
                    
                            
                    base_filter = [x[0] for x in filtered_cases]
                    project_filter = [x[1] for x in filtered_cases]
                    df_base = df_base[base_filter]
                    df_project = df_project[project_filter]
                    
                    
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
                        if col_base in dict_incremental_mapping:

                            col_project = dict_incremental_mapping[col_base]
                            df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
                    
                    if select_source == 'Base':
                        df = df_base.copy()
                    if select_source == 'Project':
                        df = df_project.copy()
                    elif select_source == 'Incremental':
                        df = df_incremental.copy()
                    
                    dfs.append(df)
            
                    # Convert index to datetime
                    df.index = dates
            
                    selected_dates_strings.append(st.select_slider(f'Select date slice {i+1}', options=[d.strftime('%Y-%m-%d') for d in dates],value=[d.strftime('%Y-%m-%d') for d in dates][-1]))
                    selected_dates_objects.append(dt.datetime.strptime(selected_dates_strings[i], "%Y-%m-%d"))
        
                    weights.append(st.number_input(f'Weight {i+1}',0,100,1))            
        
                    data = df.loc[selected_dates_objects[i]]
                    data_sorted = data.sort_values()
                                    
                    df_cumprob = pd.DataFrame(index = data_sorted.index)
                    df_cumprob['value'] = data_sorted
                    #df_cumprob['cum_prob'] = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                    df_cumprob['cum_prob'] = np.arange(len(data_sorted),0, -1) / len(data_sorted)
                    
                    dfs_cumprob.append(df_cumprob)
            
            p10_rankings = pd.DataFrame(index = df_cumprob.index)
            p50_rankings = pd.DataFrame(index = df_cumprob.index)
            p90_rankings = pd.DataFrame(index = df_cumprob.index)
            
            for i in range(num_groups):
                df_cumprob = dfs_cumprob[i]
                
                for index in p50_rankings.index:
                    p10_rankings.loc[index, str(i)+'_'+selected_props[i]] = abs((df_cumprob.loc[index,'cum_prob']-0.1))*weights[i]
                    p50_rankings.loc[index, str(i)+'_'+selected_props[i]] = abs((df_cumprob.loc[index,'cum_prob']-0.5))*weights[i]
                    p90_rankings.loc[index, str(i)+'_'+selected_props[i]] = abs((df_cumprob.loc[index,'cum_prob']-0.9))*weights[i]
                    
                    
            p10_rankings['sum'] = p10_rankings.sum(axis=1)
            p50_rankings['sum'] = p50_rankings.sum(axis=1)
            p90_rankings['sum'] = p90_rankings.sum(axis=1)
            
            p10_rankings.sort_values(by='sum',inplace=True, ascending=True)
            p50_rankings.sort_values(by='sum',inplace=True, ascending=True)
            p90_rankings.sort_values(by='sum',inplace=True, ascending=True)
                
            p10_case = p10_rankings.index[0]
            p50_case = p50_rankings.index[0]
            p90_case = p90_rankings.index[0]
            


    with c3:
        fig = make_subplots(
            rows=num_groups,
            cols=2,
            shared_xaxes=False,
            specs=[[{}, {"secondary_y": True}] for _ in range(num_groups)],
            subplot_titles=[f"{selected_identifiers[i]}: {select_source}" for i in range(num_groups) for _ in range(2)],
            horizontal_spacing=0.15
        )

        for i in range(num_groups):
            df = dfs[i]
            df_cumprob = dfs_cumprob[i]

            # Time series plot (left)
            for col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    line=dict(color='grey', width=1),
                    opacity=0.3,
                    name=col if i == 0 else None,
                    showlegend=False
                ), row=i+1, col=1)

            # Highlight P10/P50/P90
            fig.add_trace(go.Scatter(x=df.index, y=df[p90_case], line=dict(color='firebrick', width=2), name="P90"), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df[p50_case], line=dict(color='blue', width=2), name="P50"), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df[p10_case], line=dict(color='green', width=2), name="P10"), row=i+1, col=1)
            fig.add_vline(x=selected_dates_objects[i], line_color="black", line_dash="dash", row=i+1, col=1)

            # Right: Histogram (secondary y) + cumulative probability (primary y)
            fig.add_trace(go.Histogram(
                x=df_cumprob["value"],
                opacity=0.3,
                marker=dict(color='lightgrey'),
                name="Distribution",
                showlegend=(i == 0)
            ), row=i+1, col=2, secondary_y=True)

            fig.add_trace(go.Scatter(
                x=df_cumprob["value"],
                y=df_cumprob["cum_prob"],
                mode="markers",
                marker=dict(color="lightblue", size=10),
                name="Cumulative Prob" if i == 0 else None
            ), row=i+1, col=2, secondary_y=False)
            
          


            # Horizontal quantile lines
            for q, color in zip([0.1, 0.5, 0.9], ['green', 'blue', 'firebrick']):
                fig.add_hline(y=q, line_color=color, line_dash="dot", row=i+1, col=2, secondary_y=False)

            # Triangle markers for P10/P50/P90
            for case, color in zip([p10_case, p50_case, p90_case], ['green', 'blue', 'firebrick']):
                fig.add_trace(go.Scatter(
                    x=[df_cumprob.loc[case, 'value']],
                    y=[df_cumprob.loc[case, 'cum_prob']],
                    mode='markers',
                    marker=dict(color=color, size=20, symbol='triangle-up'),
                    name=f"{case}" if i == 0 else None,
                    showlegend=(i == 0)
                ), row=i+1, col=2, secondary_y=False)

            # Axis labels
            fig.update_yaxes(title_text=selected_props[i], row=i+1, col=1)
            fig.update_yaxes(title_text="Cumulative Prob", row=i+1, col=2, secondary_y=False)
            fig.update_yaxes(title_text="Histogram", row=i+1, col=2, secondary_y=True)
            fig.update_xaxes(title_text="Date", row=i+1, col=1)
            fig.update_xaxes(title_text=f"{selected_props[i]} @ {selected_dates_strings[i]}", row=i+1, col=2)

        # Layout
        fig.update_layout(
            height=plot_height * 200 * num_groups,
            title="Case Selection - Ensemble Profiles",
            showlegend=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        import pandas as pd
        from io import BytesIO

        # Store per-case yearly profiles
        p10_profiles = {}
        p50_profiles = {}
        p90_profiles = {}

        # Loop over each selected property and its dataframe
        for i in range(num_groups):
            df = dfs[i]  # Time series for that property

            # Ensure datetime index
            df.index = pd.to_datetime(df.index)

            # Resample to yearly (mean) and format index as 01-01-Y
            yearly = df.resample('YE').mean().copy()
            yearly.index = pd.to_datetime([f"01-01-{y.year}" for y in yearly.index])

            # Extract profiles for the P10, P50, and P90 cases
            p10_profiles[selected_props[i]] = yearly[p10_case]
            p50_profiles[selected_props[i]] = yearly[p50_case]
            p90_profiles[selected_props[i]] = yearly[p90_case]

        # Convert to DataFrames
        df_p10_export = pd.DataFrame(p10_profiles)
        df_p50_export = pd.DataFrame(p50_profiles)
        df_p90_export = pd.DataFrame(p90_profiles)

        # Write to Excel in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_p10_export.to_excel(writer, sheet_name="P10_Yearly")
            df_p50_export.to_excel(writer, sheet_name="P50_Yearly")
            df_p90_export.to_excel(writer, sheet_name="P90_Yearly")
        output.seek(0)

        # Show download button in Streamlit
        st.download_button(
            label="Download P10/P50/P90 Profiles (Excel)",
            data=output,
            file_name="P10_P50_P90_Yearly_Profiles.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
