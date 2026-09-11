import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent

# Page configuration
st.set_page_config(page_title="County Emissions Visualization", layout="wide")
st.title("🗺️ Michigan County Emissions Analysis")

# Load data
@st.cache_data
def load_data():
    SHAPEFILE_PATH = script_dir + "County_Boundaries_-_Extended\County_Boundaries_-_Extended.shp"
    EMISSIONS_CSV_PATH = script_dir + "aggregated_results.csv"
    EMISSIONS_DIFF_PATH = script_dir + "aggregated_results_difference.csv"
    
    gdf = gpd.read_file(SHAPEFILE_PATH)
    df_baseline = pd.read_csv(EMISSIONS_CSV_PATH)
    df_diff = pd.read_csv(EMISSIONS_DIFF_PATH)
    
    return gdf, df_baseline, df_diff

gdf, df_baseline, df_diff = load_data()

# Standardize names
def standardize_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    name = name.replace('_', ' ')
    name = name.replace('.', '')
    name = name.title()
    name = ' '.join(name.split())
    return name

gdf['county_standardized'] = gdf['Name'].apply(standardize_name)
df_baseline['county_standardized'] = df_baseline['county'].apply(standardize_name)
df_diff['county_standardized'] = df_diff['county'].apply(standardize_name)

# Convert percentage strings to floats
pollutants = ['pollutantID_2', 'pollutantID_3', 'pollutantID_87', 'pollutantID_100', 'pollutantID_110']
pollutant_names = {
    'pollutantID_2': 'CO',
    'pollutantID_3': 'NOx',
    'pollutantID_87': 'VOC',
    'pollutantID_100': 'PM10',
    'pollutantID_110': 'PM2.5'
}

for col in pollutants:
    df_diff[col] = df_diff[col].str.rstrip('%').astype(float)

# Convert baseline to numeric if needed
for col in pollutants:
    df_baseline[col] = pd.to_numeric(df_baseline[col], errors='coerce')

# Sidebar controls
st.sidebar.header("📊 Visualization Controls")

# Select view type
view_type = st.sidebar.radio(
    "Select View Type:",
    ["Baseline Emissions", "Single Pollutant by Sources", "All Pollutants Comparison", "County-Level Analysis"]
)

if view_type == "Baseline Emissions":
    st.subheader("Baseline Emissions by Pollutant")
    
    # Filter baseline data (Default source only)
    baseline_data = df_baseline[df_baseline['speed_source'] == 'Default'].copy()
    
    # Select pollutant
    selected_pollutant = st.sidebar.selectbox(
        "Select Pollutant:",
        pollutants,
        format_func=lambda x: pollutant_names[x]
    )
    
    # Create tabs for two views
    tab1, tab2 = st.tabs(["Absolute Emissions", "Emissions per Sq Mile"])
    
    with tab1:
        st.write("### 📊 Total Baseline Emission Amounts")
        
        # Merge baseline data with geometry
        merged_gdf = gdf.merge(
            baseline_data[['county_standardized', selected_pollutant]], 
            left_on='county_standardized', 
            right_on='county_standardized', 
            how='left'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        vmin = merged_gdf[selected_pollutant].quantile(0.02)
        vmax = merged_gdf[selected_pollutant].quantile(0.98)
        
        # Plot choropleth
        merged_gdf.plot(
            column=selected_pollutant,
            ax=ax,
            cmap='RdYlGn_r',
            edgecolor='black',
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax
        )
        
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Emission Amount (tons)', orientation='vertical', shrink=0.8, pad=0.05)
        
        # Add values above each county
        for idx_row, row in merged_gdf.iterrows():
            if pd.notna(row[selected_pollutant]):
                centroid = row.geometry.centroid
                value = row[selected_pollutant]
                ax.text(
                    centroid.x, centroid.y, 
                    f'{value:.1f}',
                    fontsize=6, ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
                )
        
        ax.set_title(f'{pollutant_names[selected_pollutant]} - Baseline Emission Amounts by County (tons)', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display statistics
        st.subheader("📈 Baseline Emission Statistics")
        stats_data = merged_gdf[selected_pollutant].dropna()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean (tons)", f"{stats_data.mean():.2f}")
        with col2:
            st.metric("Median (tons)", f"{stats_data.median():.2f}")
        with col3:
            st.metric("Min (tons)", f"{stats_data.min():.2f}")
        with col4:
            st.metric("Max (tons)", f"{stats_data.max():.2f}")
        with col5:
            st.metric("Std Dev", f"{stats_data.std():.2f}")
    
    with tab2:
        st.write("### 📊 Baseline Emission Rates (per Square Mile)")
        
        # Merge baseline data with geometry
        merged_gdf = gdf.merge(
            baseline_data[['county_standardized', selected_pollutant]], 
            left_on='county_standardized', 
            right_on='county_standardized', 
            how='left'
        )
        
        # Convert square meters to square miles (1 sq mile = 2.58999e6 sq meters)
        sqm_to_sqmi = 2.58999e6
        merged_gdf[f'{selected_pollutant}_per_sqmi'] = merged_gdf[selected_pollutant] / (merged_gdf['Shape__Are'] / sqm_to_sqmi)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        vmin = merged_gdf[f'{selected_pollutant}_per_sqmi'].quantile(0.02)
        vmax = merged_gdf[f'{selected_pollutant}_per_sqmi'].quantile(0.98)
        
        # Plot choropleth
        merged_gdf.plot(
            column=f'{selected_pollutant}_per_sqmi',
            ax=ax,
            cmap='RdYlGn_r',
            edgecolor='black',
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax
        )
        
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Emission Rate (tons/sq mi)', orientation='vertical', shrink=0.8, pad=0.05)
        
        # Add values above each county
        for idx_row, row in merged_gdf.iterrows():
            if pd.notna(row[f'{selected_pollutant}_per_sqmi']):
                centroid = row.geometry.centroid
                value = row[f'{selected_pollutant}_per_sqmi']
                ax.text(
                    centroid.x, centroid.y, 
                    f'{value:.2f}',
                    fontsize=6, ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
                )
        
        ax.set_title(f'{pollutant_names[selected_pollutant]} - Baseline Emission Rate by County (tons/sq mi)', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display statistics
        st.subheader("📈 Emission Rate Statistics")
        stats_data = merged_gdf[f'{selected_pollutant}_per_sqmi'].dropna()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean (tons/sq mi)", f"{stats_data.mean():.3f}")
        with col2:
            st.metric("Median (tons/sq mi)", f"{stats_data.median():.3f}")
        with col3:
            st.metric("Min (tons/sq mi)", f"{stats_data.min():.3f}")
        with col4:
            st.metric("Max (tons/sq mi)", f"{stats_data.max():.3f}")
        with col5:
            st.metric("Std Dev", f"{stats_data.std():.3f}")

elif view_type == "Single Pollutant by Sources":
    st.subheader("Single Pollutant Across Multiple Data Sources")
    
    # Select pollutant
    selected_pollutant = st.sidebar.selectbox(
        "Select Pollutant:",
        pollutants,
        format_func=lambda x: pollutant_names[x]
    )
    
    # Filter data
    df_filtered = df_diff[df_diff['speed_source'] != 'Default'].copy()
    sources = sorted(df_filtered['speed_source'].unique())[:4]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for source_idx, source in enumerate(sources):
        source_data = df_filtered[df_filtered['speed_source'] == source].copy()
        
        # Merge data
        merged_gdf = gdf.merge(
            source_data[['county_standardized', selected_pollutant]], 
            left_on='county_standardized', 
            right_on='county_standardized', 
            how='left'
        )
        
        ax = axes[source_idx]
        vmin = -25
        vmax = 25
        
        # Plot choropleth
        merged_gdf.plot(
            column=selected_pollutant,
            ax=ax,
            cmap='RdYlGn_r',
            edgecolor='black',
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax
        )
        
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Change (%)', orientation='vertical', shrink=0.8, pad=0.05)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add percentage values
        for idx_row, row in merged_gdf.iterrows():
            if pd.notna(row[selected_pollutant]):
                centroid = row.geometry.centroid
                percentage_value = row[selected_pollutant]
##                ax.text(
##                    centroid.x, centroid.y, 
##                    f'{percentage_value:.1f}%',
##                    fontsize=5, ha='center', va='center',
##                    fontweight='bold',
##                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
##                )
        
        ax.set_title(f'{source}', fontsize=13, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f'{pollutant_names[selected_pollutant]} - Percentage Change by Data Source', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

elif view_type == "All Pollutants Comparison":
    st.subheader("All Pollutants Overview")
    
    # Select data source
    df_filtered = df_diff[df_diff['speed_source'] != 'Default'].copy()
    selected_source = st.sidebar.selectbox(
        "Select Data Source:",
        sorted(df_filtered['speed_source'].unique())
    )
    
    source_data = df_filtered[df_filtered['speed_source'] == selected_source].copy()
    
    # Create figure with all pollutants
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for pollutant_idx, pollutant in enumerate(pollutants):
        merged_gdf = gdf.merge(
            source_data[['county_standardized', pollutant]], 
            left_on='county_standardized', 
            right_on='county_standardized', 
            how='left'
        )
        
        ax = axes[pollutant_idx]
        vmin = -25
        vmax = 25
        
        merged_gdf.plot(
            column=pollutant,
            ax=ax,
            cmap='RdYlGn_r',
            edgecolor='black',
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax
        )
        
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Change (%)', shrink=0.8, pad=0.05)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        ax.set_title(f'{pollutant_names[pollutant]}', fontsize=13, fontweight='bold')
        ax.axis('off')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.suptitle(f'All Pollutants - {selected_source}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

elif view_type == "County-Level Analysis":
    st.subheader("County-Level Detailed Analysis")
    
    # Select county
    counties = sorted(gdf['county_standardized'].unique())
    selected_county = st.sidebar.selectbox("Select County:", counties)
    
    # Get county data
    county_data = df_diff[df_diff['county_standardized'] == selected_county]
    
    if not county_data.empty:
        # Get baseline values
        baseline_data = df_baseline[df_baseline['county_standardized'] == selected_county]
        
        if not baseline_data.empty:
            st.write(f"### 📌 Baseline Emissions for {selected_county} County")
            baseline_values = baseline_data[pollutants].iloc[0]
            
            # Create baseline display
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric(pollutant_names['pollutantID_2'], f"{baseline_values['pollutantID_2']:.2f} tons")
            with col2:
                st.metric(pollutant_names['pollutantID_3'], f"{baseline_values['pollutantID_3']:.2f} tons")
            with col3:
                st.metric(pollutant_names['pollutantID_87'], f"{baseline_values['pollutantID_87']:.2f} tons")
            with col4:
                st.metric(pollutant_names['pollutantID_100'], f"{baseline_values['pollutantID_100']:.2f} tons")
            with col5:
                st.metric(pollutant_names['pollutantID_110'], f"{baseline_values['pollutantID_110']:.2f} tons")
        
        # Display percentage change data table
        st.write(f"### 📊 Percentage Changes by Data Source")
        display_data = county_data[['speed_source'] + pollutants].copy()
        display_data.columns = ['Data Source'] + [pollutant_names[p] for p in pollutants]
        
        # Format as percentages
        for col in [pollutant_names[p] for p in pollutants]:
            display_data[col] = display_data[col].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_data, width='stretch')
        
        # Create comparison chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        county_data_numeric = county_data[['speed_source'] + pollutants].set_index('speed_source')
        county_data_numeric.columns = [pollutant_names[p] for p in pollutants]
        
        county_data_numeric.plot(kind='bar', ax=ax, width=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Percentage Change (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Data Source', fontsize=12, fontweight='bold')
        ax.set_title(f'{selected_county} County - Pollutant Comparison by Data Source', 
                     fontsize=14, fontweight='bold')
        ax.legend(title='Pollutant', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
    else:
        st.warning(f"No data found for {selected_county}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("County Emissions Estimates in Michigan State\nBased on MOVES and Probe Speed Bin Analysis")
