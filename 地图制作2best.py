import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Set font and disable LaTeX to avoid symbol issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# Read data
df = pd.read_excel('precise_bohai_sea_data.xlsx')

# Extract coordinates from location column
def extract_coordinates(coord_str):
    """Extract latitude and longitude from string"""
    try:
        # Remove parentheses and spaces, then split
        coord_str = coord_str.strip('()').replace(' ', '')
        lon, lat = coord_str.split(',')
        return float(lon), float(lat)
    except:
        return None, None

# Extract coordinates
df[['longitude', 'latitude']] = df['处理位置'].apply(
    lambda x: pd.Series(extract_coordinates(x))
)

# Remove rows with NaN coordinates
df = df.dropna(subset=['longitude', 'latitude'])

# Calculate mean values for each location
location_means = df.groupby('处理位置').agg({
    'nppv': 'mean',
    'thetao': 'mean', 
    'kd': 'mean',
    'chl': 'mean',
    'so': 'mean',
    'spco2': 'mean',
    'longitude': 'first',
    'latitude': 'first'
}).reset_index()

print(f"Processed data contains {len(location_means)} monitoring stations")

# Set Bohai Sea map boundaries
bohai_bounds = {
    'lon_min': 117.5,
    'lon_max': 123,
    'lat_min': 37.0,
    'lat_max': 41.0
}

# Define parameters with units and color maps
parameters = [
    ('nppv', 'Net Primary Productivity (mg C/m³/day)', 'viridis'),
    ('thetao', 'Sea Temperature (°C)', 'plasma'),
    ('chl', 'Chlorophyll Concentration (mg/m³)', 'YlGn'),
    ('so', 'Salinity (PSU)', 'Blues'),
    ('spco2', 'Surface CO₂ Partial Pressure (µatm)', 'Reds'),
    ('kd', 'Diffuse Attenuation Coefficient (m⁻¹)', 'Purples')
]

# Create individual plots for each parameter
for param, title, cmap in parameters:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set map extent
    ax.set_extent([bohai_bounds['lon_min'], bohai_bounds['lon_max'], 
                   bohai_bounds['lat_min'], bohai_bounds['lat_max']], 
                  crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Add grid lines
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Get data for current parameter
    values = location_means[param]
    
    # Create scatter plot with color representing parameter values
    # and size representing relative magnitude
    sizes = 50 + 100 * (values - values.min()) / (values.max() - values.min())
    
    scatter = ax.scatter(location_means['longitude'], location_means['latitude'],
                        c=values, cmap=cmap, s=sizes, alpha=0.7,
                        edgecolors='black', linewidth=0.5)
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(param.upper())
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add Bohai Sea label
    ax.text(119.5, 40.5, 'Bohai Sea', transform=ccrs.PlateCarree(),
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Create comprehensive map - all parameters in one plot with different colors
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set map extent
ax.set_extent([bohai_bounds['lon_min'], bohai_bounds['lon_max'], 
               bohai_bounds['lat_min'], bohai_bounds['lat_max']], 
              crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Add grid lines
gl = ax.gridlines(draw_labels=True, alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Define colors for each parameter
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
param_names = ['nppv', 'thetao', 'chl', 'so', 'spco2', 'kd']
param_labels = ['NPP (mg C/m³/day)', 'Temp (°C)', 'Chl (mg/m³)', 
                'Salinity (PSU)', 'pCO₂ (µatm)', 'Kd (m⁻¹)']

# Plot all stations
for i, (param, color, label) in enumerate(zip(param_names, colors, param_labels)):
    # Normalize sizes
    values = location_means[param]
    sizes = 30 + 70 * (values - values.min()) / (values.max() - values.min())
    
    scatter = ax.scatter(location_means['longitude'], location_means['latitude'],
                        s=sizes, color=color, alpha=0.6, label=label,
                        edgecolors='black', linewidth=0.5)

# Add legend
ax.legend(loc='upper right', title='Environmental Parameters')

# Add title and labels
ax.set_title('Comprehensive Distribution of Monitoring Stations in Bohai Sea', 
             fontsize=16, fontweight='bold')
ax.text(119.5, 40.5, 'Bohai Sea', transform=ccrs.PlateCarree(),
        fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add north arrow
ax.text(117.2, 37.2, 'N', transform=ccrs.PlateCarree(),
        fontsize=12, fontweight='bold', ha='center', va='center')

plt.tight_layout()
plt.show()

# Create time series plots for representative stations
selected_locations = location_means['处理位置'].head(3)

for i, location in enumerate(selected_locations):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    location_data = df[df['处理位置'] == location].copy()
    location_data['time'] = pd.to_datetime(location_data['time'])
    location_data = location_data.sort_values('time')
    
    # Get coordinates for this station
    lon = location_data['longitude'].iloc[0]
    lat = location_data['latitude'].iloc[0]
    
    # Plot multiple parameter time series
    # Temperature
    ax.plot(location_data['time'], location_data['thetao'], 'r-', label='Temperature', linewidth=2)
    ax.set_ylabel('Temperature (°C)', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    
    # Chlorophyll (secondary axis)
    ax2 = ax.twinx()
    ax2.plot(location_data['time'], location_data['chl'], 'g-', label='Chlorophyll', linewidth=2)
    ax2.set_ylabel('Chlorophyll (mg/m³)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax.set_title(f'Monitoring Station ({lon:.3f}°E, {lat:.3f}°N) Time Series')
    ax.tick_params(axis='x', rotation=45)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

# Create statistical summary chart
fig, ax = plt.subplots(figsize=(10, 6))
params = ['NPP', 'Temperature', 'Kd', 'Chlorophyll', 'Salinity', 'pCO₂']
units = ['(mg C/m³/day)', '(°C)', '(m⁻¹)', '(mg/m³)', '(PSU)', '(µatm)']
param_labels_with_units = [f"{p}\n{u}" for p, u in zip(params, units)]
means = [location_means['nppv'].mean(), 
         location_means['thetao'].mean(),
         location_means['kd'].mean(),
         location_means['chl'].mean(),
         location_means['so'].mean(),
         location_means['spco2'].mean()]

bars = ax.bar(param_labels_with_units, means, 
              color=['skyblue', 'coral', 'lightgreen', 'gold', 'violet', 'orange'])
ax.set_ylabel('Mean Value')
ax.set_title('Mean Values of Environmental Parameters in Bohai Sea')
ax.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1 * max(means),
            f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Create correlation heatmap
correlation_data = location_means[['nppv', 'thetao', 'kd', 'chl', 'so', 'spco2']]
correlation_matrix = correlation_data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add correlation values as text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold')

# Set ticks and labels
ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.columns)))
ax.set_xticklabels(['NPP', 'Temp', 'Kd', 'Chl', 'Sal', 'pCO₂'])
ax.set_yticklabels(['NPP', 'Temp', 'Kd', 'Chl', 'Sal', 'pCO₂'])

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Correlation Coefficient')

ax.set_title('Correlation Matrix of Environmental Parameters')
plt.tight_layout()
plt.show()

# Output basic information
print("\n=== Bohai Sea Environmental Monitoring Data Summary ===")
print(f"Number of monitoring stations: {len(location_means)}")
print(f"Data time range: {df['time'].min()} to {df['time'].max()}")
print(f"Longitude range: {df['longitude'].min():.3f}°E - {df['longitude'].max():.3f}°E")
print(f"Latitude range: {df['latitude'].min():.3f}°N - {df['latitude'].max():.3f}°N")

print("\nParameter Statistics:")
stats = location_means[['nppv', 'thetao', 'kd', 'chl', 'so', 'spco2']].describe()
print(stats)