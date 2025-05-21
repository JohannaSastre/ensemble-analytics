

&nbsp;
<h4 style='margin-bottom: 0px;'>Purpose:</h4>
<hr style='margin-top: 0px; margin-bottom: 10px; border: 1px solid #ccc;'/>

The Incremental Ensemble Analysis in Streamlit app is a comprehensive visual tool for analyzing production performance differences between two simulation cases: a Base case and a Project (or modified) case. Its main goal is to help you evaluate the incremental value or effect of changes (e.g., new wells, optimization strategies, injection plans) by comparing case results across regions, wells, and properties.

To visualize and quantify the difference (increment) in reservoir production metrics over time and at specific time slices — enabling better decision-making in reservoir management, forecasting, and well planning.

&nbsp;
<h4 style="margin-bottom: 0px;">Key Functionalities:</h4>
<hr style='margin-top: 0px; margin-bottom: 10px; border: 1px solid #ccc;'/>

**Case Loading and Mapping:**

- Upload .pkl files for Base and Project cases.

- Upload a mapping .xlsx file that defines which realization in the project corresponds to which realization in the base.

- The app aligns cases by name and timestamps and stores all data in session state.

**Analysis Tab:**
Shows time series plots for:

- Base vs. Project

- Incremental (difference)

- Shows histograms at a selected date slice.

- Plots the S-curve (cumulative probability) for incremental values at the selected time slice.

- Displays P10, P50, P90 lines to evaluate uncertainty.

- Allows y-axis overrides for better comparison.



**Boxplots Tab:**
Generate boxplots across selected regions or wells.

Choose to plot:

- Base

- Project

- Incremental values

- Customize plot height and select multiple entities.

**Crossplot Tab:**

- Cross-compare two variables (X vs. Y) from Base/Project/Incremental.

- Quantile lines (P10, P50, P90) for each axis.

- Useful for identifying correlations or performance clusters.

**Waterfall Tab:**

- Aggregated view of incremental values across regions or wells.

- Shows contribution of each entity to the total difference.

- Merge minor contributors into “Other” for clarity.

- Helps explain where and why a project case differs from the base.

**Case Selection Tab:**

- Multi-property selection with weights.

- Select date slices per property.

- Rank realizations by cumulative probability (P10, P50, P90).

- Visualize both time evolution and S-curve for selected cases.

- Identifies which realizations to use for low/mid/high scenarios.

**Sales Conversions Tab:**

- Converts production volumes (oil, gas) into sales units (mstb, mmscf, mboe).

- Includes shrinkage factors and yield settings.

- Adds derived metrics like cumulative sales oil, gas, NGL, and BOE.
