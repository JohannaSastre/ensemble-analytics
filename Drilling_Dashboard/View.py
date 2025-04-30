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
        options=["Plots"],
        icons=["bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

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

# Display plots and histograms based on toggle switch
if selected == "Plots":
    show_histogram = st.checkbox("Show Histogram with S-Curve")

    if show_histogram:
        st.write("### Histogram with S-Curve")

        st.write("#### Porosity")
        plot_histogram_with_scurve(PHIT, "Porosity Distribution")

        st.write("#### Water Saturation")
        plot_histogram_with_scurve(Sw, "Water Saturation Distribution")

        st.write("#### Pressure")
        plot_histogram_with_scurve(Pe, "Pressure Distribution")

        st.write("#### Thickness")
        plot_histogram_with_scurve(Thickness, "Thickness Distribution")
    else:
        st.write("### Interactive Plots")
        st.write("#### Porosity")
        st_echarts(options=porosity_options)

        st.write("#### Water Saturation")
        st_echarts(options=water_saturation_options)

        st.write("#### Pressure")
        st_echarts(options=pressure_options)

        st.write("#### Thickness")
        st_echarts(options=thickness_options)