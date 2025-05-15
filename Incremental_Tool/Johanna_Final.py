import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import streamlit as st
import pandas as pd
import numpy as np
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


def sort_key(s):
    import re
    match = re.search(r'\d+', s)
    if match:
        return (0, int(match.group()))
    else:
        return (1, s)

########################################## Sidebar option menu
with st.sidebar:
    st.title("Incremental Value")
    
    selected = option_menu(
        menu_title="",           # Title of the sidebar
        options=["About", "Upload Data", "Sales Conversions", "Boxplots","Analysis","Crossplot","Case selection","Waterfall"
                ],  # Menu options with emojis
        icons=["info-circle","capslock", "graph-down", "bar-chart-steps","arrows-expand-vertical","diagram-3","currency-dollar","bullseye","bounding-box","clipboard-data", "gear"],  # Optional icons from Bootstrap
        default_index=0,                  # Default selected option
        )

########################################## Main menus

    
########################################## Main menus
if selected == "Extract Data":
        
    #%%
    
    def process_excel_file(df):
        #Rename columns
        col_names_new = []
        for x,y,z in zip(df.columns, df.iloc[0].values, df.iloc[1].values):
            s = x
            if str(y) != 'nan':
                s = s +' '+y
            if str(z) != 'nan':
                s = s +' '+z
            col_names_new.append(s)
        df.columns= col_names_new
        
        #Drop row 0 and 1
        df = df.iloc[2:]
        #Drop unnamed column
        for col in df.columns:
            if 'unnamed' in col.lower():
                df = df.drop(columns=[col])
        #reset index
        df.reset_index(drop=True, inplace=True)
        df['Entity Name']=[x.split(':')[1] for x in df['Entity Name']]
        return df
    
    
    prop_dict = {  "ROIP OIL_IN_PLACE SM3": 'Oil in place (sm3)',
                    "RGIP GAS_IN_PLACE SM3": 'Gas in place (sm3)',
                    "RWIP WATER_IN_PLACE SM3": 'Water in place (sm3)',
                    "ROPT OIL_PRODUCTION_CUML SM3": 'Cum oil (sm3)',
                    "RWPT WATER_PRODUCTION_CUML SM3": 'Cum water (sm3)',
                    "RGPT GAS_PRODUCTION_CUML SM3": 'Cum gas (sm3)',
                    "ROE OIL_RECOVERY_EFFICIENCY": 'Oil recovery factor (frac)',
                    "RPRH PRESSURE_HYDROCARBON_PORE_VOLUME_WEIGHTED BARSA": 'Reservoir pressure (bara)'}
    
    root_folder = st.text_input('Root folder (exclude "S:/")',value='VAL_HOD/ResX/User/ValHod_Group/Valhall/2024/Active/ResX_FFM_281024.sim')
    file_extension = st.text_input('CSV-file name',value='RegionOutput_REGION.csv')
    
         
    # 
    #  
    
    server_address = st.text_input("Server IP: Nitro-10.44.49.10, VDI-10.50.56.22",value='10.50.56.22')
    username       = st.text_input("Username:")
    password       = st.text_input("Password:", type="password")
    timeout_value = 5
    
    df_incremental_mapping_file = st.file_uploader("Upload case mapping file", type=["xlsx"])
    if df_incremental_mapping_file:
        
        df_incremental_mapping = pd.read_excel(df_incremental_mapping_file)
        dict_incremental_mapping = {df_incremental_mapping.loc[i,'Base']:df_incremental_mapping.loc[i,'Project'] for i in df_incremental_mapping.index}
        st.session_state['dict_incremental_mapping'] = dict_incremental_mapping

        
        columns = ['Base','Base filepath','Base dates','Base uploaded', 'Base dataframe',
                    'Project','Project filepath','Project uploaded', 'Project dataframe',
                    'Incremental_valid']
        dict_mapping = {i:{s:0 for s in columns} for i in df_incremental_mapping.index}            
        
        wells_master = []
        regions_master = []
        dates_master = []
        properties_master = []
        
        for i,index in enumerate(df_incremental_mapping.index):
            st.write(df_incremental_mapping.loc[index,'Base'],df_incremental_mapping.loc[index,'Project'], i+1, len(df_incremental_mapping.index)+1)
                
            for case_type in ['Base','Project']:
                dict_mapping[index][f'{case_type}'] = df_incremental_mapping.loc[index,f'{case_type}']
                filepath = root_folder+'/'+df_incremental_mapping.loc[index,f'{case_type}']+'/'+df_incremental_mapping.loc[index,f'{case_type}']+'_summary_results/'+file_extension
                
                dict_mapping[index][f'{case_type} filepath'] = filepath
                  
                
                
                 
              
    
                session = smbclient.register_session(server_address, username=username, password=password, connection_timeout=timeout_value)
                    
                path = f'{server_address}/Asset-data/{filepath}'
    
                with smbclient.open_file(path, mode='rb') as file:
                    df = pd.read_csv(file)
                    df = process_excel_file(df)
                     
                    properties = []
                    new_cols = []
                    for prop in df.columns:
                        if prop in prop_dict:
                            new_cols.append(prop_dict[prop])
                            properties.append(prop_dict[prop])
                            
                        else:
                            new_cols.append(prop)
                    df.columns = new_cols
                    
                    wells = set(df[df['Entity Type']=='WELL']['Entity Name'])
                    regions = set(df[df['Entity Type']=='REGION']['Entity Name'])
                    dates = sorted([dt.datetime.strptime(x, "%d/%b/%Y") for x in set(df['DATE'].values)])
                    
                    
                    wells_master.append(wells)
                    regions_master.append(regions)
                    dates_master.append(dates)
                    properties_master.append(properties)
                    
                    dict_mapping[index][f'{case_type} uploaded'] = True
                    dict_mapping[index][f'{case_type} dates'] = dates
                    dict_mapping[index][f'{case_type} dataframe'] = df
                        
            incremental_valid = True
            incremental_valid *= dict_mapping[index]['Base uploaded']
            incremental_valid *= dict_mapping[index]['Project uploaded']
            incremental_valid *= dict_mapping[index]['Base dates'] == dict_mapping[index]['Project dates']
            
            dict_mapping[index]['Incremental_valid'] = incremental_valid == 1
            
        
                   
        st.write()
        if all(x == wells_master[0] for x in wells_master):
            st.write('WELLS are OK')
            wells = wells_master[0]
        else:
            st.write('WARNING: There are inconsistent DATES in some realizations')
            
        if all(x == regions_master[0] for x in regions_master):
            st.write('REGIONS are OK')
            regions = regions_master[0]
        else:
            st.write('WARNING: There are inconsistent WELLS in some realizations')
            
        if all(x == dates_master[0] for x in dates_master):
            st.write('DATES are OK')
            dates = dates_master[0]
        else:
            st.write('WARNING: There are inconsistent REGIONS in some realizations')
            
        if all(x == properties_master[0] for x in properties_master):
            properties = properties_master[0]
            st.write('PROPERTIES are OK')
        else:
            st.write('WARNING: There are inconsistent PROPERTIES in some realizations')
        #%%
        cases_base = list(df_incremental_mapping['Base'])
        df_temp = pd.DataFrame(index=dates, columns = cases_base)
        data_dict_base= {'Field': {prop: df_temp.copy() for prop in properties},
                        'Wells': {well: {prop: df_temp.copy() for prop in properties} for well in wells},
                        'Regions': {region: {prop: df_temp.copy() for prop in properties} for region in regions},
                        'Metadata':{'Name':'Base',
                                    'Dates':dates,
                                    'Cases': cases_base,
                                    'Properties': properties,
                                    'Wells': wells,
                                    'Regions': regions,
                                    'Groups': {'Wells':0, 'Regions': 0},
                                    'Variables': {'Global': 0,
                                                  'Time': 0,}
                                    }
                        }
            
        cases_project = list(df_incremental_mapping['Project'])
        df_temp = pd.DataFrame(index=dates, columns = cases_project)
        data_dict_project= {'Field': {prop: df_temp.copy() for prop in properties},
                        'Wells': {well: {prop: df_temp.copy() for prop in properties} for well in wells},
                        'Regions': {region: {prop: df_temp.copy() for prop in properties} for region in regions},
                        'Metadata':{'Name':'Project',
                                    'Dates':dates,
                                    'Cases': cases_project,
                                    'Properties': properties,
                                    'Wells': wells,
                                    'Regions': regions,
                                    'Groups': {'Wells':0, 'Regions': 0},
                                    'Variables': {'Global': 0,
                                                  'Time': 0,}
                                    }
                        }
        #%%
        
        
        for key in dict_mapping:
            if dict_mapping[key]['Incremental_valid']:
                
                df_base = dict_mapping[key]['Base dataframe'].copy()
                case_base = dict_mapping[key]['Base']
                
                df_project = dict_mapping[key]['Project dataframe'].copy()
                case_project = dict_mapping[key]['Project']
                
                for region in regions:
                    df_base_temp = df_base[df_base['Entity Name']==region]
                    df_project_temp = df_project[df_project['Entity Name']==region]
                    
                    for prop in properties:
                        data_dict_base['Regions'][region][prop][case_base] = df_base_temp[prop].values
                        data_dict_project['Regions'][region][prop][case_project] = df_project_temp[prop].values
                     
        st.session_state['data_dict_base'] = data_dict_base
        st.session_state['data_dict_project'] = data_dict_project
    
    
        st.write(data_dict_base)
       
           
########################################## Main menus
from pathlib import Path

# Point directly to the markdown file in the current directory
ABOUT = Path("about.md")

if selected == "About":
    st.markdown("<h3 style='text-align: left; font-weight: bold;'>Incremental Dashboard</h3>", unsafe_allow_html=True)
    try:
        with open("about.md", "r", encoding="utf-8") as file:
            about_content = file.read()
            st.markdown(about_content, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("The 'about.md' file is missing. Please ensure it's in the same folder as your script.")

if selected == "Upload Data":
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.header("Base case")
        file = st.file_uploader(label="Upload Base Case File", key='1', type=['pkl'], label_visibility="hidden")
        if file:
            data_dict_base = pd.read_pickle(file)
            st.session_state['data_dict_base'] = data_dict_base

        if 'data_dict_base' in st.session_state:
            data_dict_base = st.session_state['data_dict_base']

            st.write(data_dict_base['Metadata']['Name'])
            st.write(data_dict_base['Metadata']['Cases'])
            st.write(data_dict_base['Metadata']['Properties'])
            st.write(data_dict_base['Metadata']['Dates'])

    with col2:
        st.header("Incremental Case")
        file2 = st.file_uploader(label="Upload Incremental Case File", key='2', type=['pkl'], label_visibility="hidden")
        if file2:
            data_dict_project = pd.read_pickle(file2)
            st.session_state['data_dict_project'] = data_dict_project

        if 'data_dict_project' in st.session_state:
            data_dict_project = st.session_state['data_dict_project']

            st.write(data_dict_project['Metadata']['Name'])
            st.write(data_dict_project['Metadata']['Cases'])
            st.write(data_dict_project['Metadata']['Properties'])
            st.write(data_dict_project['Metadata']['Dates'])

    with col3:
        st.header("Upload Incremental Mapping")
        file3 = st.file_uploader(label="Upload Incremental Mapping File", key='3', type=['xlsx'], label_visibility="hidden")
        if file3:
            df_incremental_mapping = pd.read_excel(file3)
            dict_incremental_mapping = {df_incremental_mapping.loc[i, 'Base']: df_incremental_mapping.loc[i, 'Project'] for i in df_incremental_mapping.index}

            st.session_state['dict_incremental_mapping'] = dict_incremental_mapping

        if 'dict_incremental_mapping' in st.session_state:
            dict_incremental_mapping = st.session_state['dict_incremental_mapping']
            st.write('Base : Project')
            st.write(dict_incremental_mapping)
######################################################### Main menus##############################################################################
if selected == "Sales Conversions":
        
    data_dict_base = st.session_state['data_dict_base']
    data_dict_project = st.session_state['data_dict_project']
    
    regions = data_dict_base['Metadata']['Regions']
    regions = sorted(regions, key=sort_key)
    props = data_dict_base['Metadata']['Properties']
    dates = data_dict_base['Metadata']['Dates']
    dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 
    
    col1,col2,col3 =st.columns(3)
    with col1:

        st.header("Unit conversions")
           
        st.number_input("Oil Yield from Gas",value = 0.022, format="%.3f")
        st.number_input("Gas shrinkage to Oil",value = 0.006, format="%.3f")
        st.number_input("Gas Shrinkage to NGL",value = 0.8793, format="%.4f")
        st.number_input("NGL yield from Sales Gas",value = 0.046, format="%.4f")
        st.number_input("stb per sm3",value = 6.293, format="%.4f")
        st.number_input("sm3 o.e oil per sm3 gas",value = 0.001, format="%.4f")
        
        
    with col2:
        st.write("Sales Oil = Oil(stb) + Gas(stb) x (Oil yield from gas)")
    
    for region in regions:
        oil = data_dict_base['Regions'][region]['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce')  
        gas = data_dict_base['Regions'][region]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce')  
        
        oil_mstb = oil * 6.293 / 1000
        gas_mmscf = gas * 35.3147 / 1e6
        
        oil_sales = oil_mstb + gas_mmscf * 0.022        
        gas_sales = gas_mmscf*0.8793 - oil_sales*0.006
        ngl_sales = gas_sales*0.046
        boe_sales = oil_sales + gas_sales/5.6226 + ngl_sales
        
        data_dict_base['Regions'][region]['Cum sales oil (mstb)'] = oil_sales
        data_dict_base['Regions'][region]['Cum sales gas (mmscf)'] = gas_sales
        data_dict_base['Regions'][region]['Cum sales NGL(mstb)'] = ngl_sales
        data_dict_base['Regions'][region]['Cum sales O.E. (mboe)'] = boe_sales

        
    for region in regions:
        oil = data_dict_project['Regions'][region]['Cum oil (sm3)'].copy().apply(pd.to_numeric, errors='coerce')  
        gas = data_dict_project['Regions'][region]['Cum gas (sm3)'].copy().apply(pd.to_numeric, errors='coerce')  
        
        oil_mstb = oil * 6.293 / 1000
        gas_mmscf = gas * 35.3147 / 1e6
        
        oil_sales = oil_mstb + gas_mmscf * 0.022             
        gas_sales = gas_mmscf*0.8793 - oil_sales*0.006
        ngl_sales = gas_sales*0.046
        boe_sales = oil_sales + gas_sales/5.6226 + ngl_sales
        
        data_dict_project['Regions'][region]['Cum sales oil (mstb)'] = oil_sales
        data_dict_project['Regions'][region]['Cum sales gas (mmscf)'] = gas_sales
        data_dict_project['Regions'][region]['Cum sales NGL(mstb)'] = ngl_sales
        data_dict_project['Regions'][region]['Cum sales O.E. (mboe)'] = boe_sales

    props = data_dict_base['Metadata']['Properties'] + ['Cum sales oil (mstb)','Cum sales gas (mmscf)','Cum sales NGL(mstb)','Cum sales O.E. (mboe)']
    props = list(set(props))
    data_dict_base['Metadata']['Properties'] = props

        
    st.write(data_dict_base['Metadata']['Properties'])



##################################################################################### ANALYSIS ############################################################################    
elif selected == "Analysis":

    ########### setup widgets
    
    data_dict_base = st.session_state['data_dict_base']
    data_dict_project = st.session_state['data_dict_project']
    
    dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 
    
    regions = data_dict_base['Metadata']['Regions']
    regions = sorted(regions, key=sort_key)
    props = data_dict_base['Metadata']['Properties']
    dates = data_dict_base['Metadata']['Dates']
    wells = data_dict_base['Metadata']['Wells']
    
    with st.sidebar.expander("Plot settings"):
        
        plot_height = st.number_input("Plot height",4,16,8)
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
    
    select_category = st.sidebar.selectbox(f"Select Category",options = ['Field','Regions','Wells'])

    if select_category == 'Field':
        selected_identifier = 'Field'
        selected_property = st.sidebar.selectbox('Select property', props)
    
        df_base = data_dict_base['Field'][selected_property].apply(pd.to_numeric, errors='coerce')  
        df_project = data_dict_project['Field'][selected_property].apply(pd.to_numeric, errors='coerce')
        
    if select_category == 'Regions':
        selected_identifier = st.sidebar.selectbox('Select region', regions)
        selected_property = st.sidebar.selectbox('Select property', props)
    
        df_base = data_dict_base['Regions'][selected_identifier][selected_property].apply(pd.to_numeric, errors='coerce')  
        df_project = data_dict_project['Regions'][selected_identifier][selected_property].apply(pd.to_numeric, errors='coerce') 
        
    if select_category == 'Wells':
        selected_identifier = st.sidebar.selectbox('Select well', wells)
        selected_property = st.sidebar.selectbox('Select property', props)
    
        df_base = data_dict_base['Wells'][selected_identifier][selected_property].apply(pd.to_numeric, errors='coerce')  
        df_project = data_dict_project['Wells'][selected_identifier][selected_property].apply(pd.to_numeric, errors='coerce')  
        
        
    df_incremental = pd.DataFrame(index=dates)
    for col_base in df_base.columns:
        col_project = dict_incremental_mapping[col_base]
        df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]

    
    selected_date = st.sidebar.select_slider('Select date',options = dates, format_func=lambda date: date.strftime("%Y-%m-%d"))
    
    base_slice = df_base.loc[selected_date].fillna(0)
    project_slice =  df_project.loc[selected_date].fillna(0)
    incremental_slice =  df_incremental.loc[selected_date].fillna(0)
    incremental_slice_sorted = incremental_slice.sort_values()
                    
    df_incremental_slice_cumprob = pd.DataFrame(index = incremental_slice_sorted.index)
    df_incremental_slice_cumprob['value'] = incremental_slice_sorted
    df_incremental_slice_cumprob['cum_prob'] = np.arange(len(incremental_slice_sorted),0, -1) / len(incremental_slice_sorted)
    
    
    ########### plot data ANALYSIS UPDATED###############################################################################################
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    # Create 2x2 subplots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Base/Project Time Series", "Base/Project Histogram",
                                        "Incremental Time Series", "Incremental Histogram + S-Curve"),
                        specs=[[{}, {}], [{}, {"secondary_y": True}]])

    # ---- Bins for consistent histograms
    bins = np.histogram_bin_edges(np.concatenate([base_slice, project_slice]), bins=30)

    # ---- Top left: Time series (base + project)
    if plot_base:
        for col in df_base.columns:
            fig.add_trace(go.Scatter(x=df_base.index, y=df_base[col], mode='lines',
                                    name=f"Base: {col}", line=dict(color='grey', width=1), opacity=0.5),
                        row=1, col=1)

    if plot_project:
        for col in df_project.columns:
            fig.add_trace(go.Scatter(x=df_project.index, y=df_project[col], mode='lines',
                                    name=f"Project: {col}", line=dict(color='blue', width=1), opacity=0.5),
                        row=1, col=1)

    fig.add_vline(x=selected_date, line_dash='dash', line_color='black', row=1, col=1)

    # ---- Top right: Histogram (base + project)
    if plot_base:
        fig.add_trace(go.Histogram(x=base_slice, name="Base", marker_color='grey', opacity=0.75,
                                xbins=dict(start=bins[0], end=bins[-1], size=(bins[1] - bins[0])),marker_line=dict(width=2.5, color='black')),
                    row=1, col=2)

    if plot_project:
        fig.add_trace(go.Histogram(x=project_slice, name="Project", marker_color='blue', opacity=0.5,
                                xbins=dict(start=bins[0], end=bins[-1], size=(bins[1] - bins[0])),marker_line=dict(width=2.5, color='black')),
                    row=1, col=2)

    # ---- Bottom left: Incremental time series
    if plot_incremental:
        for col in df_incremental.columns:
            fig.add_trace(go.Scatter(x=df_incremental.index, y=df_incremental[col], mode='lines',
                                    name=f"Incremental: {col}", line=dict(color='red'), opacity=0.5),
                        row=2, col=1)

        fig.add_vline(x=selected_date, line_dash='dash', line_color='black', row=2, col=1)

    # ---- Bottom right: Incremental histogram + S-curve
    if plot_incremental:
        fig.add_trace(go.Histogram(x=incremental_slice, name="Incremental", marker_color='red', opacity=0.5),
                    row=2, col=2, secondary_y=False)

    if plot_scurve:
        fig.add_trace(go.Scatter(x=df_incremental_slice_cumprob['value'],
                                y=df_incremental_slice_cumprob['cum_prob'],
                                mode='markers',
                                name="Cumulative Probability",
                                marker=dict(color='red', line=dict(color='black', width=1))),
                    row=2, col=2, secondary_y=True)

        # Add P10, P50, P90 lines
        for q, color in zip([0.1, 0.5, 0.9], ['firebrick', 'blue', 'green']):
            val = incremental_slice.quantile(q)
            fig.add_vline(x=val, line_dash="dashdot", line_color=color, row=2, col=2)
            fig.add_hline(y=q, line_dash="dashdot", line_color=color, row=2, col=2, secondary_y=True)

    # ---- Override axes if needed
    if override_axis_incremental:
        fig.update_yaxes(range=[incremental_yaxis_min, incremental_yaxis_max], row=2, col=1)
        fig.update_xaxes(range=[incremental_yaxis_min, incremental_yaxis_max], row=2, col=2)

    # ---- Axis labels
    fig.update_xaxes(title_text="Time (days)", row=1, col=1)
    fig.update_xaxes(title_text=f"{selected_property} @ {selected_date.strftime('%Y-%m-%d')}", row=1, col=2)
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text=f"{selected_property} @ {selected_date.strftime('%Y-%m-%d')}", row=2, col=2)

    fig.update_yaxes(title_text=selected_property, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text=selected_property, row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2, secondary_y=True)

    # ---- Layout
    fig.update_layout(
        height=plot_height * 100,
        title=dict(
            text=f"{selected_identifier}: {selected_property}",
            font=dict(size=25),  # Title font
            x=0.0,
            xanchor='left'
        ),
        showlegend=True,
        template="plotly_white",
        font=dict(size=30),  # Global font size (axes, ticks, legend)
        barmode='overlay'
   )

    # ---- Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

         
            
    with tab2:
        
        st.write(f'Base - {selected_identifier} - {selected_property}')
        st.dataframe(df_base,use_container_width=True)
        st.write(f'Project - {selected_identifier} - {selected_property}')
        st.dataframe(df_project,use_container_width=True)
        st.write(f'Incremental - {selected_identifier} - {selected_property}')
        st.dataframe(df_incremental,use_container_width=True)
        
 #################################################################### Crossplot #########################################       
elif selected == "Crossplot":

    ########### setup widgets########################################Update#########################################################
    
    data_dict_base = st.session_state['data_dict_base']
    data_dict_project = st.session_state['data_dict_project']
    
    regions = data_dict_base['Metadata']['Regions']
    regions = sorted(regions, key=sort_key)
    props = data_dict_base['Metadata']['Properties']
    dates = data_dict_base['Metadata']['Dates']
    wells = data_dict_base['Metadata']['Wells']
    
    dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 
    
    with st.sidebar.expander("Plot settings"):
        plot_height = st.number_input("Plot height",4,16,8)
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
    
    with st.sidebar.expander("X-Axis variable:",expanded=True):
        
        x_selected_category = st.selectbox('Select category (X)', options = ['Field','Region','Well'])
        x_selected_type = st.selectbox('Select source (X)',['Base','Project','Incremental'])
        x_selected_property = st.selectbox('Select property (X)', props)

        if x_selected_category == 'Field':
            x_selected_identifier = 'Field'
            x_df_base = data_dict_base['Field'][x_selected_property].apply(pd.to_numeric, errors='coerce')  
            x_df_project = data_dict_project['Field'][x_selected_property].apply(pd.to_numeric, errors='coerce')  
            
        if x_selected_category == 'Region':
            x_selected_identifier = st.selectbox('Select region (X)', regions)
            x_df_base = data_dict_base['Regions'][x_selected_identifier][x_selected_property].apply(pd.to_numeric, errors='coerce')  
            x_df_project = data_dict_project['Regions'][x_selected_identifier][x_selected_property].apply(pd.to_numeric, errors='coerce')  
            
        if x_selected_category == 'Well':
            x_selected_identifier = st.selectbox('Select well (X)', wells)
            x_df_base = data_dict_base['Wells'][x_selected_identifier][x_selected_property].apply(pd.to_numeric, errors='coerce')  
            x_df_project = data_dict_project['Wells'][x_selected_identifier][x_selected_property].apply(pd.to_numeric, errors='coerce')  
        
        #Calculate incremental
        x_df_incremental = pd.DataFrame(index=dates)
        for col_base in x_df_base.columns:
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
            
    with st.sidebar.expander("X-Axis variable:",expanded=True):
        
        y_selected_category = st.selectbox('Select category (Y)', options = ['Field','Region','Well'])
        y_selected_type = st.selectbox('Select source (Y)',['Base','Project','Incremental'])
        y_selected_property = st.selectbox('Select property (Y)', props)

        if y_selected_category == 'Field':
            y_selected_identifier = 'Field'
            y_df_base = data_dict_base['Field'][y_selected_property].apply(pd.to_numeric, errors='coerce')  
            y_df_project = data_dict_project['Field'][y_selected_property].apply(pd.to_numeric, errors='coerce')  
            
        if y_selected_category == 'Region':
            y_selected_identifier = st.selectbox('Select region (Y)', regions)
            y_df_base = data_dict_base['Regions'][y_selected_identifier][y_selected_property].apply(pd.to_numeric, errors='coerce')  
            y_df_project = data_dict_project['Regions'][y_selected_identifier][y_selected_property].apply(pd.to_numeric, errors='coerce')  
            
        if y_selected_category == 'Well':
            y_selected_identifier = st.selectbox('Select well (Y)', wells)
            y_df_base = data_dict_base['Wells'][y_selected_identifier][y_selected_property].apply(pd.to_numeric, errors='coerce')  
            y_df_project = data_dict_project['Wells'][y_selected_identifier][y_selected_property].apply(pd.to_numeric, errors='coerce')  
        
        #Calculate incremental
        y_df_incremental = pd.DataFrame(index=dates)
        for col_base in y_df_base.columns:
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
    
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
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


############################################## BOXPLOTS UPDATED #################################################################################################
        

elif selected == "Boxplots":
    tab1, tab2 = st.tabs(['Plots', 'Data'])
    with tab1:
        ########### setup widgets
        data_dict_base = st.session_state['data_dict_base']
        data_dict_project = st.session_state['data_dict_project']

        regions = data_dict_base['Metadata']['Regions']
        regions = sorted(regions, key=sort_key)
        props = data_dict_base['Metadata']['Properties']
        dates = data_dict_base['Metadata']['Dates']
        wells = data_dict_base['Metadata']['Wells']

        dict_incremental_mapping = st.session_state['dict_incremental_mapping']

        with st.sidebar.expander("Plot settings"):
            plot_height = st.number_input("Plot height", 4, 16, 8)

        selected_category = st.sidebar.selectbox("Select category", options=["Regions", "Wells"])
        selected_plot_type = st.sidebar.selectbox('Select ensemble', ["Base", "Project", "Incremental"])
        selected_property = st.sidebar.selectbox('Select property', props)
        selected_date = st.sidebar.select_slider('Select date', options=dates, format_func=lambda date: date.strftime("%Y-%m-%d"))

        box_data_base = pd.DataFrame()
        box_data_project = pd.DataFrame()
        box_data_incremental = pd.DataFrame()

        if selected_category == 'Regions':
            selected_identifiers = st.sidebar.multiselect("Selected regions", regions, regions)
            for region in selected_identifiers:
                df_base = data_dict_base['Regions'][region][selected_property].apply(pd.to_numeric, errors='coerce')
                df_project = data_dict_project['Regions'][region][selected_property].apply(pd.to_numeric, errors='coerce')

                df_incremental = pd.DataFrame(index=dates)
                for col_base in df_base.columns:
                    col_project = dict_incremental_mapping[col_base]
                    df_incremental[col_project + ' - ' + col_base] = df_project[col_project] - df_base[col_base]

                base_slice = df_base.loc[selected_date]
                project_slice = df_project.loc[selected_date]
                incremental_slice = df_incremental.loc[selected_date]

                box_data_base[region] = base_slice
                box_data_project[region] = project_slice
                box_data_incremental[region] = incremental_slice

        if selected_category == 'Wells':
            selected_identifiers = st.sidebar.multiselect("Selected wells", wells, wells)
            for well in selected_identifiers:
                df_base = data_dict_base['Wells'][well][selected_property].apply(pd.to_numeric, errors='coerce')
                df_project = data_dict_project['Wells'][well][selected_property].apply(pd.to_numeric, errors='coerce')

                df_incremental = pd.DataFrame(index=dates)
                for col_base in df_base.columns:
                    col_project = dict_incremental_mapping[col_base]
                    df_incremental[col_project + ' - ' + col_base] = df_project[col_project] - df_base[col_base]

                base_slice = df_base.loc[selected_date]
                project_slice = df_project.loc[selected_date]
                incremental_slice = df_incremental.loc[selected_date]

                box_data_base[well] = base_slice
                box_data_project[well] = project_slice
                box_data_incremental[well] = incremental_slice

        if selected_plot_type == 'Base':
            df = box_data_base.copy()
        elif selected_plot_type == 'Project':
            df = box_data_project.copy()
        elif selected_plot_type == 'Incremental':
            df = box_data_incremental.copy()
###################################################################################################BoxPlot Updated###########################################
        # Plotly boxplot logic
        import plotly.graph_objects as go

        if not df.empty:
            fig = go.Figure()

            for col in df.columns:
                values = df[col].dropna().values
                if len(values) > 0:
                    fig.add_trace(go.Box(
                        y=values,
                        name=col,
                        marker_color='darkblue',
                        boxmean=False,
                        hovertemplate=f"<b>{col}</b><br>Value: %{{y}}<extra></extra>"
                    ))

            fig.update_layout(
                title=dict(
                    text=f"Boxplot of {selected_property} by {selected_category} ({selected_plot_type})",
                    font=dict(size=26)  # Title font size
                ),
                xaxis=dict(
                    title=dict(
                        text=selected_category,
                        font=dict(size=18)
                    ),
                    tickfont=dict(size=14),
                    tickangle=45  # Optional: tilt labels for readability
                ),
                yaxis=dict(
                    title=dict(
                        text=f"{selected_property} @ {selected_date.strftime('%Y-%m-%d')}",
                        font=dict(size=18)
                    ),
                    tickfont=dict(size=14)
                ),
                height=plot_height * 100,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for plotting.")

    with tab2:
        st.dataframe(box_data_base, use_container_width=True)
        st.dataframe(box_data_project, use_container_width=True)
        st.dataframe(box_data_incremental, use_container_width=True)

#################################################################### Waterfall #########################################
elif selected == "Waterfall":
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
        # Setup
        data_dict_base = st.session_state['data_dict_base']
        data_dict_project = st.session_state['data_dict_project']
        dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 

        regions = sorted(data_dict_base['Metadata']['Regions'], key=sort_key)
        props = data_dict_base['Metadata']['Properties']
        dates = data_dict_base['Metadata']['Dates']
        wells = list(data_dict_base['Metadata']['Wells'])

        with st.sidebar.expander("Plot settings"):
            plot_height = st.number_input("Plot height", 4, 16, 8)
            merge_zeros = st.checkbox('Merge values', value=True)
            if merge_zeros:
                cutoff = st.number_input("Cutoff value", value=1.0)

        selected_category = st.sidebar.selectbox("Select category", options=["Regions", "Wells"])
        selected_property = st.sidebar.selectbox("Select property", props)
        selected_date = st.sidebar.select_slider("Select date", options=dates, format_func=lambda d: d.strftime("%Y-%m-%d"))

        with st.sidebar.expander("Plot content"):
            show_base = st.checkbox("Show Base", value=True)
            show_project = st.checkbox("Show Project", value=True)
            show_other = st.checkbox("Show Other", value=True)

        # Identifiers and selection
        identifiers = regions if selected_category == 'Regions' else wells

        with st.sidebar.expander("Select identifiers to display"):
            identifiers_selected = st.multiselect("Choose identifiers", identifiers, default=identifiers)

        box_data_base = pd.DataFrame()
        box_data_project = pd.DataFrame()
        box_data_incremental = pd.DataFrame()

        for name in identifiers:
            if selected_category == 'Regions':
                df_base = data_dict_base['Regions'][name][selected_property].apply(pd.to_numeric, errors='coerce')
                df_project = data_dict_project['Regions'][name][selected_property].apply(pd.to_numeric, errors='coerce')
            else:
                df_base = data_dict_base['Wells'][name][selected_property].apply(pd.to_numeric, errors='coerce')
                df_project = data_dict_project['Wells'][name][selected_property].apply(pd.to_numeric, errors='coerce')

            df_incremental = pd.DataFrame(index=dates)
            for col_base in df_base.columns:
                col_project = dict_incremental_mapping[col_base]
                df_incremental[col_project + ' - ' + col_base] = df_project[col_project] - df_base[col_base]

            box_data_base[name] = df_base.loc[selected_date]
            box_data_project[name] = df_project.loc[selected_date]
            box_data_incremental[name] = df_incremental.loc[selected_date]

        # Aggregate
        base = box_data_base.mean().sum()
        project = box_data_project.mean().sum()
        incremental = box_data_incremental.mean()

        df_temp = pd.DataFrame(index=identifiers)
        df_temp["Value"] = list(incremental)
        df_temp = df_temp.loc[identifiers_selected]

        # Waterfall logic
        if merge_zeros:
            df_main = df_temp[df_temp["Value"].abs() >= cutoff]
            df_other = df_temp[df_temp["Value"].abs() < cutoff]
        else:
            df_main = df_temp
            df_other = pd.DataFrame(columns=["Value"])

        values = []
        labels = []
        measures = []

        if show_base:
            values.append(base)
            labels.append("Base")
            measures.append("absolute")

        for name, row in df_main.iterrows():
            values.append(row["Value"])
            labels.append(name)
            measures.append("relative")

        if show_other and not df_other.empty:
            values.append(df_other["Value"].sum())
            labels.append("Other")
            measures.append("relative")

        if show_project:
            values.append(project)
            labels.append("Project")
            measures.append("total")

        # Plotly Waterfall Chart
        fig = go.Figure(go.Waterfall(
            name="Incremental",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=[f"{v / 1e6:.2f}M" for v in values],  
            textposition="outside",
        ))

        fig.update_layout(
            title=f"Waterfall Chart for {selected_property} @ {selected_date.strftime('%Y-%m-%d')}",
            height=plot_height * 100,
            showlegend=False,
            template="plotly_white",
            font=dict(size=14),
            margin=dict(t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(df_temp, use_container_width=True)
        st.dataframe(box_data_base, use_container_width=True)
        st.dataframe(box_data_project, use_container_width=True)
        st.dataframe(box_data_incremental, use_container_width=True)

        
#################################################################### CASE SELECTION        #########################################
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime
    import numpy as np

elif selected == "Case selection":
    data_dict_base = st.session_state['data_dict_base']
    data_dict_project = st.session_state['data_dict_project']
    dict_incremental_mapping = st.session_state['dict_incremental_mapping']

    regions = sorted(data_dict_base['Metadata']['Regions'], key=sort_key)
    props = data_dict_base['Metadata']['Properties']
    dates = data_dict_base['Metadata']['Dates']
    wells = list(data_dict_base['Metadata']['Wells'])

    with st.sidebar.expander("Plot settings"):
        plot_height = st.number_input("Plot height", 1, 16, 4)
    num_groups = st.sidebar.number_input('Number of properties:', 1, 10, 1)
    select_source = st.sidebar.selectbox("Select Source", ['Base', 'Project', 'Incremental'])

    selected_identifiers, selected_props, selected_dates_objects = [], [], []
    dfs, dfs_cumprob, weights = [], [], []

    for i in range(num_groups):
        with st.sidebar.expander(f"Property {i+1}", expanded=True):
            cat = st.selectbox(f"Select Category {i+1}", ['Field', 'Region', 'Well'], key=f"cat_{i}")
            prop = st.selectbox(f"Select Property {i+1}", props, key=f"prop_{i}")
            selected_props.append(prop)

            if cat == 'Field':
                df_base = data_dict_base['Field'][prop].apply(pd.to_numeric, errors='coerce')
                df_project = data_dict_project['Field'][prop].apply(pd.to_numeric, errors='coerce')
                selected_identifiers.append('Field')
            elif cat == 'Region':
                region = st.selectbox(f"Select Region {i+1}", regions, key=f"region_{i}")
                df_base = data_dict_base['Regions'][region][prop].apply(pd.to_numeric, errors='coerce')
                df_project = data_dict_project['Regions'][region][prop].apply(pd.to_numeric, errors='coerce')
                selected_identifiers.append(region)
            else:
                well = st.selectbox(f"Select Well {i+1}", wells, key=f"well_{i}")
                df_base = data_dict_base['Wells'][well][prop].apply(pd.to_numeric, errors='coerce')
                df_project = data_dict_project['Wells'][well][prop].apply(pd.to_numeric, errors='coerce')
                selected_identifiers.append(well)

            df_incremental = pd.DataFrame(index=dates)
            for col_base in df_base.columns:
                if col_base in dict_incremental_mapping:
                    col_project = dict_incremental_mapping[col_base]
                    if col_project in df_project.columns:
                        df_incremental[f"{col_project} - {col_base}"] = df_project[col_project] - df_base[col_base]

            df = {"Base": df_base, "Project": df_project, "Incremental": df_incremental}[select_source]
            df.index = dates
            dfs.append(df)


            selected_date_str = st.select_slider(f"Select Date Slice {i+1}", options=[d.strftime('%Y-%m-%d') for d in dates], key=f"date_{i}")
            selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d")
            selected_dates_objects.append(selected_date)

            series = df.loc[selected_date]
            df_cum = pd.DataFrame({'value': series.sort_values()})
            df_cum['cum_prob'] = np.linspace(1, 0, len(df_cum))
            dfs_cumprob.append(df_cum)

            weights.append(st.number_input(f"Weight {i+1}", 0, 100, 1, key=f"weight_{i}"))

    # Ranking
    p10, p50, p90 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i, df_cum in enumerate(dfs_cumprob):
        for idx in df_cum.index:
            p10.loc[idx, f"{i}_{selected_props[i]}"] = abs(df_cum.loc[idx, 'cum_prob'] - 0.9) * weights[i]
            p50.loc[idx, f"{i}_{selected_props[i]}"] = abs(df_cum.loc[idx, 'cum_prob'] - 0.5) * weights[i]
            p90.loc[idx, f"{i}_{selected_props[i]}"] = abs(df_cum.loc[idx, 'cum_prob'] - 0.1) * weights[i]

    p10['sum'] = p10.sum(axis=1)
    p50['sum'] = p50.sum(axis=1)
    p90['sum'] = p90.sum(axis=1)

    p10_case, p50_case, p90_case = p10.sort_values('sum').index[0], p50.sort_values('sum').index[0], p90.sort_values('sum').index[0]

    
    # Define subplot titles in the correct order (left to right, top to bottom)
    subplot_titles = []
    for i in range(num_groups):
        subplot_titles.append(f"{selected_props[i]} - {selected_identifiers[i]} (Time Series)")
        subplot_titles.append(f"{selected_props[i]} - {selected_identifiers[i]} (S-Curve)")

    # Create the subplot figure
    fig = make_subplots(
        rows=num_groups,
        cols=2,
        subplot_titles=[f"{selected_props[i]} - {selected_identifiers[i]} (Time Series)" if j % 2 == 0
                        else f"{selected_props[i]} - {selected_identifiers[i]} (S-Curve)"
                        for i in range(num_groups) for j in range(2)],
        shared_xaxes=False,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        specs=[[{}, {"secondary_y": True}] for _ in range(num_groups)]
     )
    

    for i in range(num_groups):
        df = dfs[i]
        df_cum = dfs_cumprob[i]
        date = selected_dates_objects[i]

        for col in df.columns:
            if col == p90_case:
                color = 'green'; width = 3; opacity = 1; show = True
            elif col == p50_case:
                color = 'blue'; width = 3; opacity = 1; show = True
            elif col == p10_case:
                color = 'red'; width = 3; opacity = 1; show = True
            else:
                color = 'lightgrey'; width = 1; opacity = 0.4; show = False

            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode='lines',
                name=col,
                showlegend=show,
                line=dict(color=color, width=width),
                opacity=opacity
            ), row=i + 1, col=1)

        # Vertical line for selected date
        fig.add_trace(go.Scatter(
            x=[date, date],
            y=[df.min().min(), df.max().max()],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Selected Date',
            showlegend=(i == 0)  # Only show in first subplot
        ), row=i + 1, col=1)
       
       # Histogram
        fig.add_trace(go.Histogram(
            x=df_cum['value'],
            name='Histogram',
            marker=dict(color='lightblue'),
            opacity=0.2,
            showlegend=(i == 0)
        ),  row=i + 1, col=2, secondary_y=True)
       
        # CDF plot
        fig.add_trace(go.Scatter(
            x=df_cum['value'],
            y=df_cum['cum_prob'],
            mode='markers',
            name='CDF',
            marker=dict(color='grey'),
            showlegend=(i == 0)  # Show legend only once
        ), row=i + 1, col=2, secondary_y=False)
   
            

        # Triangle markers for P90, P50, P10
        for case, color, label in zip([p90_case, p50_case, p10_case], ['green', 'blue', 'red'], ['P90 Case', 'P50 Case', 'P10 Case']):
            fig.add_trace(go.Scatter(
                x=[df_cum.loc[case, 'value']],
                y=[df_cum.loc[case, 'cum_prob']],
                mode='markers+text',
                name=label,
                text=[label],
                textposition="top right",
                marker=dict(size=15, color=color, symbol='triangle-up', line=dict(width=2)),
                showlegend=(i == 0)  # Legend shown once
            ), row=i + 1, col=2, secondary_y=False)

       
    for i in range(num_groups):
        fig.update_yaxes(title_text="Cumulative Probability", row=i+1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Count (Histogram)", row=i+1, col=2, secondary_y=True)
        fig.update_yaxes(showgrid=True, row=i+1, col=2, secondary_y=False)  # Keep gridlines for CDF
        fig.update_yaxes(showgrid=False, row=i+1, col=2, secondary_y=True)  # Turn OFF for histogram

    fig.update_layout(
        height=400 * num_groups,
        title_text="Case Selection Results",
        showlegend=True,
        barmode='overlay',
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Yearly Profile Summary
    with st.expander("Yearly Profile Summary"):
        st.subheader("Yearly Profile for Selected Property")

        # Select one property/identifier combination to profile
        selected_index = st.selectbox(
            "Select which property to show yearly profile for:",
            options=list(enumerate(zip(selected_props, selected_identifiers))),
            format_func=lambda x: f"{x[1][0]} - {x[1][1]}"
        )

        prop_index = selected_index[0]
        selected_prop = selected_props[prop_index]
        selected_identifier = selected_identifiers[prop_index]
        df = dfs[prop_index]

        # Select year range
        st.write("Select year range to display:")
        min_year = df.index.min().year
        max_year = df.index.max().year
        year_range = st.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

        # Compute yearly profiles for P10/P50/P90
        yearly_profiles = {}
        for label, case in zip(['P10', 'P50', 'P90'], [p10_case, p50_case, p90_case]):
            if case in df.columns:
                yearly_series = df[case].resample('Y').last()
                yearly_profiles[label] = yearly_series

        # Combine into one DataFrame
        df_yearly = pd.DataFrame(yearly_profiles)
        df_yearly.index = df_yearly.index.year
        df_yearly = df_yearly.loc[year_range[0]:year_range[1]]

        st.write(f"**{selected_prop} - {selected_identifier}**")
        st.dataframe(df_yearly, use_container_width=True)




