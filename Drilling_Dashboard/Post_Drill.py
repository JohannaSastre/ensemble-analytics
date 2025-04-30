import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from streamlit_option_menu import option_menu
import os
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Correct the logo path
logo_path = "C:\\Users\\johsas\\OneDrive - Aker BP\\python_projects\\Project_PlotDrill\\python\\Akerbp.svg"

# Load the Excel file
file_path = 'C:\\Users\\johsas\\OneDrive - Aker BP\\python_projects\\Project_PlotDrill\\Welldata.xlsx'
data = pd.read_excel(file_path)

# Clean column names (optional)
data.columns = data.columns.str.strip()


# Extract columns (update these names based on the actual column names)
x = data['MD']  # Replace 'MD' with the correct column name
PHIT = data['PHIT']  # Replace 'PHIT' with the correct column name
Sw = data['SW']  # Replace 'SW' with the correct column name
Pe = data['Pressure']  # Replace 'Pe' with the correct column name
Thickness = data['Thickness']  # Replace 'Thickness' with the correct column name

# Sidebar with logo and navigation options
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    selected = option_menu(
        menu_title="Menu",
        options=["About", "Upload Data", "Pre-Drill Phase"],
        icons=["info-circle", "cloud-upload", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "About":
    # Make the title bigger and bold
    st.markdown("<h2 style='text-align: left; font-weight: bold;'>Well Planning and Monitoring Dashboard</h2>", unsafe_allow_html=True)
    
    # Simplified relative path to the about file
    about_file_path = "about.md"
    
    # Check if the file exists
    if os.path.exists(about_file_path):
        # Read and display the content of the file with larger text
        with open(about_file_path, "r") as file:
            about_content = file.read()
            st.markdown(f"<div style='font-size: 18px;'>{about_content}</div>", unsafe_allow_html=True)
    else:
        st.error("The 'about.txt' file is missing. Please add it to the project directory.")

# Preprocess the X-axis data to ensure no decimals
x_rounded = [round(val) for val in x]  # Round all X-axis values to integers

# --- Interactive Plot 1: Porosity ---
porosity_options = {
    "tooltip": {
        "trigger": "axis",
        "axisPointer": {
            "type": "cross",
            "lineStyle": {
                "color": "#aaa",
                "width": 1,
                "type": "dashed"
            },
            "link": [{"xAxisIndex": "all"}]
        },
        "formatter": "Well Length: {b}<br>Porosity: {c}"
    },
    "xAxis": {
        "type": "category",
        "data": x_rounded,
        "name": "Well Length (mMD)",
        "nameLocation": "middle",
        "nameTextStyle": {"fontSize": 14, "padding": [10, 0, 0, 0]}
    },
    "yAxis": {
        "type": "value",
        "name": "Porosity",
        "axisLabel": {"formatter": "{value}"}
    },
    "dataZoom": [
        {
            "type": "inside",
            "yAxisIndex": 0,
            "filterMode": "filter"
        }
    ],
    "series": [{
        "name": "Porosity",
        "type": "line",
        "data": PHIT.tolist(),
        "itemStyle": {"color": "red"}
    }],
}

# --- Interactive Plot 2: Water Saturation ---
water_saturation_options = {
    "tooltip": {
        "trigger": "axis",
        "axisPointer": {
            "type": "cross",
            "lineStyle": {
                "color": "#aaa",
                "width": 1,
                "type": "dashed"
            },
            "link": [{"xAxisIndex": "all"}]
        },
        "formatter": "Well Length: {b}<br>Sw: {c}"
    },
    "xAxis": {
        "type": "category",
        "data": x_rounded,
        "name": "Well Length (mMD)",
        "nameLocation": "middle",
        "nameTextStyle": {"fontSize": 14, "padding": [10, 0, 0, 0]}
    },
    "yAxis": {
        "type": "value",
        "name": "Water Saturation",
        "axisLabel": {"formatter": "{value}"}
    },
    "dataZoom": [
        {
            "type": "inside",
            "yAxisIndex": 0,
            "filterMode": "filter"
        }
    ],
    "series": [{
        "name": "Water Saturation",
        "type": "line",
        "data": Sw.tolist(),
        "itemStyle": {"color": "blue"}
    }],
}

# --- Interactive Plot 3: Pressure ---
pressure_options = {
    "tooltip": {
        "trigger": "cross",
        "axisPointer": {
            "type": "cross",
            "lineStyle": {
                "color": "#aaa",
                "width": 1,
                "type": "dashed"
            }
        },
        "formatter": "Well Length: {b}<br>Pressure: {c}"
    },
    "xAxis": {
        "type": "category",
        "data": x_rounded,
        "name": "Well Length (mMD)",
        "nameLocation": "middle",
        "nameTextStyle": {"fontSize": 14, "padding": [10, 0, 0, 0]}
    },
    "yAxis": {
        "type": "value",
        "name": "Pressure (bar)",
        "axisLabel": {"formatter": "{value}"}
    },
    "dataZoom": [
        {
            "type": "inside",
            "yAxisIndex": 0,
            "filterMode": "filter"
        }
    ],
    "series": [{
        "name": "Pressure",
        "type": "line",
        "data": Pe.tolist(),
        "itemStyle": {"color": "green"}
    }],
}

# --- Interactive Plot 4: Thickness ---
thickness_options = {
    "tooltip": {
        "trigger": "axis",
        "axisPointer": {
            "type": "cross",
            "lineStyle": {
                "color": "#aaa",
                "width": 1,
                "type": "dashed"
            }
        },
        "formatter": "Well Length: {b}<br>Thickness: {c}"
    },
    "xAxis": {
        "type": "category",
        "data": x_rounded,
        "name": "Well Length (mMD)",
        "nameLocation": "middle",
        "nameTextStyle": {"fontSize": 14, "padding": [10, 0, 0, 0]}
    },
    "yAxis": {
        "type": "value",
        "name": "Thickness (m)",
        "axisLabel": {"formatter": "{value}"}
    },
    "dataZoom": [
        {
            "type": "inside",
            "yAxisIndex": 0,
            "filterMode": "filter"
        }
    ],
    "series": [{
        "name": "Thickness",
        "type": "line",
        "data": Thickness.tolist(),
        "itemStyle": {"color": "orange"}
    }],
}

def calculate_statistics(data):
    """Calculate P10, P50, P90 statistics for a given dataset."""
    p10 = np.percentile(data, 10)
    p50 = np.percentile(data, 50)
    p90 = np.percentile(data, 90)
    return p10, p50, p90

def plot_histogram_with_scurve(data, title):
    """Plot a histogram with S-curve for the given dataset."""
    # Calculate P10, P50, P90
    p10, p50, p90 = calculate_statistics(data)
    data_sorted = np.sort(data)
    cum_prob = np.arange(len(data_sorted), 0, -1) / len(data_sorted)

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(20, 4))
    
    # Plot the histogram
    ax1.hist(data, bins=30, edgecolor='black', alpha=0.7, label='Histogram')
    ax1.set_title(title)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')

    
# Add grids to the plot
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot the S-curve
    ax2 = ax1.twinx()
    ax2.plot(data_sorted, cum_prob, color='red', label='S-curve')
    ax2.set_ylabel('Cumulative Probability')

    # Highlight P10, P50, P90 as points on the S-curve
    points = [p10, p50, p90]
    probabilities = [0.9, 0.5, 0.1]
    labels = ['P10', 'P50', 'P90']
    colors = ['green', 'blue', 'firebrick']

    for point, prob, label, color in zip(points, probabilities, labels, colors):
        ax2.scatter(point, prob, color=color, zorder=5)  # Plot the point
        ax2.text(point, prob, f"{label}: {point:.2f}", color=color, fontsize=10, ha='left', va='bottom')  # Add label

    # Add legend
    fig.legend(loc='upper left')
    
    # Render the plot in Streamlit
    st.pyplot(fig)
    plt.close()

# Display plots and histograms based on dropdown menu
if selected == "Pre-Drill Phase":
    # Create columns to position the dropdown menu to the right
    col1, col2 = st.columns([3, 1])  # Adjust column proportions (3:1)

    with col2:  # Place the dropdown in the right column
        # Dropdown menu for selecting the type of plot
        plot_type = st.selectbox(
            "Select Plot Type",
            options=["Interactive Plots", "Histogram with S-Curve"],
            key="plot_type_dropdown"
        )

    if plot_type == "Interactive Plots":
        st.write("### Interactive Plots")
        st.write("#### Porosity")
        st_echarts(options=porosity_options)

        st.write("#### Water Saturation")
        st_echarts(options=water_saturation_options)

        st.write("#### Pressure")
        st_echarts(options=pressure_options)

        st.write("#### Thickness")
        st_echarts(options=thickness_options)

    elif plot_type == "Histogram with S-Curve":
        st.write("### Histogram with S-Curve")

        st.write("#### Porosity")
        plot_histogram_with_scurve(PHIT, "Porosity Distribution")

        st.write("#### Water Saturation")
        plot_histogram_with_scurve(Sw, "Water Saturation Distribution")

        st.write("#### Pressure")
        plot_histogram_with_scurve(Pe, "Pressure Distribution")
