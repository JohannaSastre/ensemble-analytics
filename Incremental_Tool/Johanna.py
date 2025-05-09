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
from streamlit_echarts import st_echarts
import numpy as np
import pandas as pd

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
        options=["Upload Data", "Sales Conversions", "Boxplots","Analysis","Crossplot","Case selection","Waterfall"
                ],  # Menu options with emojis
        icons=["capslock", "graph-down", "bar-chart-steps","arrows-expand-vertical","diagram-3","currency-dollar","bullseye","bounding-box","clipboard-data", "gear"],  # Optional icons from Bootstrap
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
       ############# Main menus
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
    
    ########### plot data
    
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
        fig,axs = plt.subplots(2,2,figsize=(18,plot_height))
        fig.suptitle(selected_identifier+': '+selected_property)
        
        ax1 = axs[0,0]
        ax2 = axs[0,1]
        ax3 = axs[1,0]
        ax4 = axs[1,1]
        
        bins = np.histogram_bin_edges(np.concatenate([base_slice, project_slice]), bins=30)
    
        if plot_base:
            ax1.plot(df_base,color='grey',alpha = 0.5)
            ax1.plot([],[],color='grey',label='Base')
            ax2.hist(base_slice, color = 'grey', edgecolor = 'black', bins= bins, label = 'Base',alpha = 0.5)
        
        if plot_project:
            ax1.plot(df_project,color='blue',alpha = 0.5)
            ax1.plot([],[],color='blue',label='Project')  
            ax2.hist(project_slice, color = 'blue', edgecolor = 'black', bins= bins, label = 'Project',alpha = 0.5)
    
        if plot_incremental:
            ax3.plot(df_incremental,color='red',alpha = 0.5)
            ax3.plot([],[],color='red',label='Incremental')   
            ax4.hist(incremental_slice, color = 'red', edgecolor = 'black', bins= 30, label = 'Incremental',alpha = 0.5)
        
        if plot_scurve:
            ax4_s = ax4.twinx()
            
            
            ax4_s.scatter(df_incremental_slice_cumprob['value'], df_incremental_slice_cumprob['cum_prob'], color='red', edgecolor='black', alpha=0.9, label="Cumulative Probability")
            ax4_s.axhline(0.1, color='green', linestyle='dashdot')
            ax4_s.axhline(0.5, color='blue', linestyle='dashdot')
            ax4_s.axhline(0.9, color='firebrick', linestyle='dashdot')            
            
            ax4_s.axvline(incremental_slice.quantile(0.9), color='green', linestyle='dashdot')
            ax4_s.axvline(incremental_slice.quantile(0.5), color='blue', linestyle='dashdot')
            ax4_s.axvline(incremental_slice.quantile(0.1), color='firebrick', linestyle='dashdot')
                        
    
        ax1.axvline(selected_date, color='black', linestyle='dashed', label="Selected Timeslice",zorder=3)
        ax3.axvline(selected_date, color='black', linestyle='dashed', label="Selected Timeslice",zorder=3)
    
    
        for ax in [ax2,ax4]:
            ax.grid()
            ax.legend()
            ax.set_xlabel(f'{selected_property} @ {selected_date.strftime("%Y-%m-%d")}')
        for ax in [ax1,ax3]:
            ax.grid()
            ax.legend()
            ax.set_xlabel('Time (days)')
            ax.set_ylabel(selected_property)
            
            
            
        if override_axis_incremental:
            ax3.set_ylim(incremental_yaxis_min,incremental_yaxis_max)
            ax4.set_xlim(incremental_yaxis_min,incremental_yaxis_max)
    
        st.pyplot(fig,use_container_width=True)           
            
    with tab2:
        
        st.write(f'Base - {selected_identifier} - {selected_property}')
        st.dataframe(df_base,use_container_width=True)
        st.write(f'Project - {selected_identifier} - {selected_property}')
        st.dataframe(df_project,use_container_width=True)
        st.write(f'Incremental - {selected_identifier} - {selected_property}')
        st.dataframe(df_incremental,use_container_width=True)
        
        
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

#############################################BOXPLOTS

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
                    df_incremental[col_project+' - '+col_base] = df_project[col_project] - df_base[col_base]
                
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
                    df_incremental[col_project+' - '+col_base] = df_project[col_project] - df_base[col_base]
                
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

        # Prepare data for echarts
        categories = df.columns.tolist()  # X-axis: Selected category (Regions or Wells)
        box_data = []
        tooltip_data = []  # Precompute tooltip content

    with tab2:
        st.dataframe(box_data_base, use_container_width=True)           
        st.dataframe(box_data_project, use_container_width=True)           
        st.dataframe(box_data_incremental, use_container_width=True)


        # Prepare data for echarts
    categories = df.columns.tolist()  # X-axis: Selected category (Regions or Wells)
    box_data = []

    
    #for col in df.columns:
        #values = df[col].dropna().values   # <-- divided here
    for col in df.columns:
        values = df[col].dropna().values

        if selected_property in ["Oil in place (sm3)", "Gas in place (sm3)", "Water in place (sm3)"]:
            values = values / 1e8
        elif selected_property in ["Cum Oil (sm3)", "Cum Water (sm3)", "Cum Gas (sm3)"]:
            values = values / 1e7

        min_val = float(np.min(values))
        p10 = float(np.percentile(values, 10))
        median = float(np.median(values))
        p90 = float(np.percentile(values, 90))
        max_val = float(np.max(values))
        box_data.append([min_val, p10, median, p90, max_val])

    from streamlit_echarts import JsCode
    from streamlit_echarts import st_echarts, JsCode
    # ECharts options
    options = {
    "title": {
        "text": f"Boxplot of {selected_property} by {selected_category}",
        "left": "left",
    },
    "tooltip": {
        "trigger": "item",
        "confine": True,
        "axisPointer": {
         "type": "cross"
        }
    },
    "xAxis": {
        "type": "category",
        "data": categories,
        "name": selected_category,
        "nameLocation": "middle",
        "nameGap": 30,
        "axisLabel": {"formatter": "{value}"},
    },
    "yAxis": {
        "type": "value",
        "name": f"{selected_property}",
        "nameLocation": "middle",
        "nameGap": 50,
        "axisLabel": {"formatter": "{value}"},
        "splitLine": {
            "lineStyle": {"color": "#d3d3d3", "width": 1},
        },
        "minorSplitLine": {
            "show": True,
            "lineStyle": {"color": "#e8e8e8", "width": 0.5},
        },
    },
    "grid": {
        "bottom": 100,
        "backgroundColor": "#ffffff",
    },
    #"legend": {
        #"selected": {"Boxplot": True},
    #},
    "dataZoom": [
        {"type": "inside"},
        {"type": "slider", "height": 20},
    ],
    "series": [
        {
            "name": "Boxplot",
            "type": "boxplot",
            "data": box_data,
            "itemStyle": {"color": "#b8c5f2"},
        }
    ],
}

    # Render the chart
    echarts_height = 1000
    st_echarts(options=options, height=f"{echarts_height}px")



    with tab2:
        st.dataframe(box_data_base, use_container_width=True)           
        st.dataframe(box_data_project, use_container_width=True)           
        st.dataframe(box_data_incremental, use_container_width=True)

####################################################################Waterfall#########################################
elif selected == "Waterfall":
    
    
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
    ########### setup widgets
        
        data_dict_base = st.session_state['data_dict_base']
        data_dict_project = st.session_state['data_dict_project']
        
        regions = data_dict_base['Metadata']['Regions']
        regions = sorted(regions, key=sort_key)
        props = data_dict_base['Metadata']['Properties']
        dates = data_dict_base['Metadata']['Dates']
        wells = list(data_dict_base['Metadata']['Wells'])
                                                     
        dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 

        with st.sidebar.expander("Plot settings"):
            plot_height = st.number_input("Plot height",4,16,8)
            merge_zeros = st.checkbox('Merge values', props)
            if merge_zeros:
                cutoff = st.number_input("Cutoff value",value = 1)


        selected_category = st.sidebar.selectbox("Select category",options=["Regions","Wells"])
        selected_property = st.sidebar.selectbox('Select property', props)
        selected_date = st.sidebar.select_slider('Select date',options = dates, format_func=lambda date: date.strftime("%Y-%m-%d"))
    
        box_data_base = pd.DataFrame()
        box_data_project = pd.DataFrame()
        box_data_incremental = pd.DataFrame()
        
        if selected_category == 'Regions':
            identifiers = regions
            for region in identifiers:
                df_base = data_dict_base['Regions'][region][selected_property].apply(pd.to_numeric, errors='coerce')  
                df_project = data_dict_project['Regions'][region][selected_property].apply(pd.to_numeric, errors='coerce')  
                
                df_incremental = pd.DataFrame(index=dates)
                for col_base in df_base.columns:
                    col_project = dict_incremental_mapping[col_base]
                    df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
                
                base_slice = df_base.loc[selected_date]
                project_slice =  df_project.loc[selected_date]
                incremental_slice =  df_incremental.loc[selected_date]
                
                box_data_base[region] = base_slice
                box_data_project[region] = project_slice
                box_data_incremental[region] = incremental_slice
                
        if selected_category == 'Wells':
            identifiers = wells
            for well in identifiers:
                df_base = data_dict_base['wells'][well][selected_property].apply(pd.to_numeric, errors='coerce')  
                df_project = data_dict_project['wells'][well][selected_property].apply(pd.to_numeric, errors='coerce')  
                
                df_incremental = pd.DataFrame(index=dates)
                for col_base in df_base.columns:
                    col_project = dict_incremental_mapping[col_base]
                    df_incremental[col_project+' - '+col_base] = df_project[col_project]-df_base[col_base]
                
                base_slice = df_base.loc[selected_date]
                project_slice =  df_project.loc[selected_date]
                incremental_slice =  df_incremental.loc[selected_date]
                
                box_data_base[well] = base_slice
                box_data_project[well] = project_slice
                box_data_incremental[well] = incremental_slice
            
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
        
        fig,ax = plt.subplots(figsize=(18,plot_height))
        ax.bar(df_waterfall["Category"], 
               df_waterfall["Value"], bottom=df_waterfall["Bottom"],
               color=df_waterfall["Color"], edgecolor="black")
        
        for i, value in enumerate(df_waterfall.index):
            height = df_waterfall['Total'][i]
            value = df_waterfall['Value'][i]
            ax.text(i, height*1.001, f"{value:.3e}", ha="center", va="bottom", rotation = 45)

        # Formatting
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=30)
          
        ax.set_ylabel(f'{selected_property} @ {selected_date.strftime("%Y-%m-%d")}')
        ax.set_xlabel(selected_category)
        ax.set_ylim(df_waterfall["Total"].min()*0.95,df_waterfall["Total"][1:-1].max()*1.05)
        ax.grid()
        ax.legend()
        plt.title('Waterfall chart for MEAN Incremental values')
        
        st.pyplot(fig, use_container_width=True)
    
    
    with tab2:            
        st.dataframe(df_waterfall,use_container_width=True,height=700)          
        st.dataframe(box_data_base,use_container_width=True)           
        st.dataframe(box_data_project,use_container_width=True)           
        st.dataframe(box_data_incremental,use_container_width=True)           
        
################################################################################Case Selection##############################################

elif selected == "Case selection":
    
    # Retrieve base data from session state
    data_dict_base = st.session_state['data_dict_base']
    data_dict_project = st.session_state['data_dict_project']
    
    regions = data_dict_base['Metadata']['Regions']
    regions = list(sorted(regions, key=sort_key))
    props = data_dict_base['Metadata']['Properties']
    dates = data_dict_base['Metadata']['Dates']
    wells = list(data_dict_base['Metadata']['Wells'])
                                         
    dict_incremental_mapping = st.session_state['dict_incremental_mapping'] 




    # Allow the user to select the number of expander groups
    #num_groups = st.sidebar.number_input("Select number of variable sets (maximum 3)", min_value=1, max_value=3, value=1)
    
    with st.sidebar.expander("Plot settings"):
        plot_height = st.number_input("Plot height",1,16,4)
    num_groups = st.sidebar.number_input('Number of properties:',1,10,1)
    select_source = st.sidebar.selectbox(f"Select Source",options = ['Base','Project','Incremental'])

    selected_identifiers = []
    selected_props = []
    selected_dates_objects = []
    selected_dates_strings = []
    weights = []
    
    p50_cases = pd.DataFrame()
    
    dfs = []
    dfs_cumprob = []
    
    tab1, tab2 = st.tabs(['Plots','Data'])
    with tab1:
        # Create a 3-column, 2-row subplot grid (for now..., remember: num_groups is hardcoded and set to 1)
        fig, axs = plt.subplots(num_groups,2, figsize=(12, plot_height*num_groups))
        
        for i in range(num_groups):
            with st.sidebar:
                with st.expander(f"Property {i+1}", expanded=True):
                
                    select_category = st.selectbox(f"Select Category {i+1}",options = ['Field','Region','Well'])
                    selected_props.append(st.selectbox(f'Property {i+1}', props, index=min(i,len(props)-1)))
    
                    if select_category == 'Field':
                        selected_identifiers.append('Field')
                        selected_identifier = selected_identifiers[i]
                        
                        df_base = data_dict_base['Field'][selected_props[i]].apply(pd.to_numeric, errors='coerce')           
                        df_project = data_dict_project['Field'][selected_props[i]].apply(pd.to_numeric, errors='coerce')   
                        
                    if select_category == 'Region':
                        selected_identifiers.append(st.selectbox(f'Select region {i+1}', regions))
                        selected_identifier = selected_identifiers[i]
                        
                        df_base = data_dict_base['Regions'][selected_identifier][selected_props[i]].apply(pd.to_numeric, errors='coerce')           
                        df_project = data_dict_project['Regions'][selected_identifier][selected_props[i]].apply(pd.to_numeric, errors='coerce')   
                        
                    if select_category == 'Wells':
                        selected_identifiers.append(st.selectbox(f'Select well {i+1}', wells))
                        selected_identifier = selected_identifiers[i]
                        
                        df_base = data_dict_base['Wells'][selected_identifier][selected_props[i]].apply(pd.to_numeric, errors='coerce')           
                        df_project = data_dict_project['Wells'][selected_identifier][selected_props[i]].apply(pd.to_numeric, errors='coerce')   
                        
                        
                        
                    df_incremental = pd.DataFrame(index=dates)
                    for col_base in df_base.columns:
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
                p10_rankings.loc[index, str(i)+'_'+selected_props[i]] = abs((df_cumprob.loc[index,'cum_prob']-0.9))*weights[i]
                p50_rankings.loc[index, str(i)+'_'+selected_props[i]] = abs((df_cumprob.loc[index,'cum_prob']-0.5))*weights[i]
                p90_rankings.loc[index, str(i)+'_'+selected_props[i]] = abs((df_cumprob.loc[index,'cum_prob']-0.1))*weights[i]
                
                
        p10_rankings['sum'] = p10_rankings.sum(axis=1)
        p50_rankings['sum'] = p50_rankings.sum(axis=1)
        p90_rankings['sum'] = p90_rankings.sum(axis=1)
        
        p10_rankings.sort_values(by='sum',inplace=True, ascending=True)
        p50_rankings.sort_values(by='sum',inplace=True, ascending=True)
        p90_rankings.sort_values(by='sum',inplace=True, ascending=True)
            
        p10_case = p10_rankings.index[0]
        p50_case = p50_rankings.index[0]
        p90_case = p90_rankings.index[0]
        
        
        for i in range(num_groups):
            
            df = dfs[i]
            df_cumprob = dfs_cumprob[i]
            
            if num_groups == 1:
                ax = axs[0]
            else:
                ax = axs[i,0]
                
            ax.plot(df,color='grey', alpha=0.5)    
            ax.axvline(selected_dates_objects[i], color='black', linestyle='dashed', label="Selected Timeslice", zorder=3)
            ax.plot(df[p90_case], color='firebrick', linestyle='solid', label=p90_case)
            ax.plot(df[p50_case], color='blue', linestyle='solid', label=p50_case)
            ax.plot(df[p10_case], color='green', linestyle='solid', label=p10_case)
            ax.set_xlabel('Date')
            ax.set_ylabel(selected_props[i])
            ax.legend()
            ax.set_title(selected_identifiers[i]+': '+select_source)
            ax.grid()
            
            
            if num_groups == 1:
                ax1 = axs[1]
            else:
                ax1 = axs[i,1]
                
            ax1.scatter(df_cumprob['value'], df_cumprob['cum_prob'], color='lightblue', alpha=0.9, label="Cumulative Probability")
            ax1.axhline(0.1, color='Firebrick', linestyle='dashdot')
            ax1.axhline(0.5, color='blue', linestyle='dashdot')
            ax1.axhline(0.9, color='green', linestyle='dashdot')
                            
            ax1.scatter(df_cumprob.loc[p90_case,'value'],df_cumprob.loc[p90_case,'cum_prob'], color='firebrick', s=150, label=p90_case, edgecolor='black', linewidth=1, marker='^')
            ax1.scatter(df_cumprob.loc[p50_case,'value'],df_cumprob.loc[p50_case,'cum_prob'], color='blue', s=150, label=p50_case, edgecolor='black', linewidth=1, marker='^')
            ax1.scatter(df_cumprob.loc[p10_case,'value'],df_cumprob.loc[p10_case,'cum_prob'], color='green', s=150, label=p10_case, edgecolor='black', linewidth=1, marker='^' )
            
            ax1.set_xlabel(selected_props[i] +' @ '+ selected_dates_strings[i])
            ax1.set_ylabel("Cumulative Probability")
            ax1.set_title(selected_identifiers[i]+': '+select_source)
            #ax1.invert_yax1is()
            ax1.grid(True)
            ax1.legend()
    
        # Adjust layout
        plt.tight_layout()
        # Display in Streamlit
        st.pyplot(fig, use_container_width=True)
    
    with tab2:
        st.write('P10 rankings')
        st.write(p10_rankings)
        st.write('P50 rankings')
        st.write(p50_rankings)
        st.write('P90 rankings')
        st.write(p90_rankings)    
    
        st.write('Base data')
        st.dataframe(df_base)
        st.write('Project data')
        st.dataframe(df_project)        
        st.write('Incremental data')
        st.dataframe(df_incremental)