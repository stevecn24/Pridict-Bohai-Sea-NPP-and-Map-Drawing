import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Set font to avoid mathematical symbol issues
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a font that supports math symbols
plt.rcParams['mathtext.fontset'] = 'stix'     # Use STIX fonts for math symbols
plt.rcParams['font.size'] = 14  # Increased font size for larger maps

# Set figure DPI to 600 for high resolution
plt.rcParams['figure.dpi'] = 600

# Read data
df = pd.read_excel('precise_bohai_sea_data.xlsx')

# Extract coordinates from location column
def extract_coordinates(coord_str):
    """Extract longitude and latitude from string"""
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

print(f"Processed data contains {len(location_means)} monitoring points")

# Set Bohai Sea map boundaries
bohai_bounds = {
    'lon_min': 117.0,
    'lon_max': 126.0,
    'lat_min': 37.0,
    'lat_max': 41.0
}

# Create multi-subplot for different parameters - INCREASED FIGURE SIZE
fig = plt.figure(figsize=(24, 18))  # Increased from (20, 15)

# Define parameters and corresponding color maps
parameters = [
    ('nppv', 'Net Primary Productivity (NPP)', 'viridis'),
    ('thetao', 'Sea Temperature (°C)', 'plasma'),
    ('chl', 'Chlorophyll Concentration', 'YlGn'),
    ('so', 'Salinity', 'Blues'),
    ('spco2', 'Sea Surface CO$_2$ Partial Pressure', 'Reds'),
    ('kd', 'Diffuse Attenuation Coefficient', 'Purples')
]

for i, (param, title, cmap) in enumerate(parameters):
    ax = fig.add_subplot(2, 3, i+1, projection=ccrs.PlateCarree())
    
    # Set map extent
    ax.set_extent([bohai_bounds['lon_min'], bohai_bounds['lon_max'], 
                   bohai_bounds['lat_min'], bohai_bounds['lat_max']], 
                  crs=ccrs.PlateCarree())
    
    # Add map features with enhanced visibility
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=2)  # Thicker coastline
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.2, zorder=2)
    
    # Add grid lines with better visibility
    gl = ax.gridlines(draw_labels=True, alpha=0.7, linewidth=0.8)  # Increased alpha and linewidth
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12}  # Larger grid labels
    gl.ylabel_style = {'size': 12}
    
    # Get current parameter data
    values = location_means[param]
    
    # Create scatter plot with SMALLER points for better clarity
    sizes = 20 + 50 * (values - values.min()) / (values.max() - values.min())  # Reduced sizes
    
    scatter = ax.scatter(location_means['longitude'], location_means['latitude'],
                        c=values, cmap=cmap, s=sizes, alpha=0.8,  # Increased alpha
                        edgecolors='black', linewidth=0.5, zorder=3)  # Thinner borders
    
    # Add color bar with larger font
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(title.split('(')[0].strip(), fontsize=14)
    cbar.ax.tick_params(labelsize=12)  # Larger tick labels
    
    # Add title with larger font
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)  # Increased font size and padding
    
    # Add Bohai Sea label with larger font
    ax.text(119.5, 40.5, 'Bohai Sea', transform=ccrs.PlateCarree(),
            fontsize=14, ha='center', va='center',  # Increased font size
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, linewidth=1.2))

plt.suptitle('Spatial Distribution of Environmental Monitoring Points in Bohai Sea', 
             fontsize=22, fontweight='bold', y=0.95)  # Increased font size
plt.tight_layout()
plt.savefig('bohai_environmental_parameters_small.png', dpi=600, bbox_inches='tight')
plt.show()

# Create comprehensive map - all parameters in one plot with different colors - LARGER FIGURE
fig = plt.figure(figsize=(16, 14))  # Increased from (12, 10)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set map extent
ax.set_extent([bohai_bounds['lon_min'], bohai_bounds['lon_max'], 
               bohai_bounds['lat_min'], bohai_bounds['lat_max']], 
              crs=ccrs.PlateCarree())

# Add map features with enhanced visibility
ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=2)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.2, zorder=2)

# Add grid lines with better visibility
gl = ax.gridlines(draw_labels=True, alpha=0.7, linewidth=0.8)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

# Create colors for each parameter
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
param_names = ['nppv', 'thetao', 'chl', 'so', 'spco2', 'kd']
param_labels = ['NPP', 'Temperature', 'Chlorophyll', 'Salinity', 'CO$_2$', 'Kd']

# Plot all points with SMALLER sizes
for i, (param, color, label) in enumerate(zip(param_names, colors, param_labels)):
    # Normalize sizes with smaller base values
    values = location_means[param]
    sizes = 10 + 30 * (values - values.min()) / (values.max() - values.min())  # Reduced sizes
    
    scatter = ax.scatter(location_means['longitude'], location_means['latitude'],
                        s=sizes, color=color, alpha=0.7, label=label,  # Increased alpha
                        edgecolors='black', linewidth=0.5, zorder=3)  # Thinner borders

# Add legend with larger font
ax.legend(loc='upper right', title='Environmental Parameters', fontsize=12, title_fontsize=13)

# Add title and label with larger fonts
ax.set_title('Comprehensive Distribution of Environmental Monitoring Points in Bohai Sea', 
             fontsize=18, fontweight='bold', pad=15)  # Increased font size and padding
ax.text(119.5, 40.5, 'Bohai Sea', transform=ccrs.PlateCarree(),
        fontsize=14, ha='center', va='center',  # Increased font size
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, linewidth=1.2))

# Add scale and north arrow with larger font
ax.text(117.2, 37.2, 'N', transform=ccrs.PlateCarree(),
        fontsize=14, fontweight='bold', ha='center', va='center')  # Increased font size

plt.tight_layout()
plt.savefig('bohai_comprehensive_distribution_small.png', dpi=600, bbox_inches='tight')
plt.show()

# Create static snapshot for time series animation (select 3 representative points) - LARGER FIGURE
selected_locations = location_means['处理位置'].head(3)

fig, axes = plt.subplots(3, 1, figsize=(16, 12))  # Increased from (12, 10)

for i, location in enumerate(selected_locations):
    location_data = df[df['处理位置'] == location].copy()
    location_data['time'] = pd.to_datetime(location_data['time'])
    location_data = location_data.sort_values('time')
    
    # Get coordinates for this point
    lon = location_data['longitude'].iloc[0]
    lat = location_data['latitude'].iloc[0]
    
    # Plot multiple parameter time series
    ax = axes[i]
    
    # Temperature with thicker line
    ax.plot(location_data['time'], location_data['thetao'], 'r-', label='Temperature', linewidth=2.5)  # Thicker line
    ax.set_ylabel('Temperature (°C)', color='red', fontsize=13)  # Larger font
    ax.tick_params(axis='y', labelcolor='red', labelsize=11)  # Larger tick labels
    
    # Chlorophyll (secondary axis)
    ax2 = ax.twinx()
    ax2.plot(location_data['time'], location_data['chl'], 'g-', label='Chlorophyll', linewidth=2.5)  # Thicker line
    ax2.set_ylabel('Chlorophyll', color='green', fontsize=13)  # Larger font
    ax2.tick_params(axis='y', labelcolor='green', labelsize=11)  # Larger tick labels
    
    ax.set_title(f'Monitoring Point ({lon:.3f}°E, {lat:.3f}°N) Time Series', fontsize=14)  # Larger title
    ax.tick_params(axis='x', rotation=45, labelsize=11)  # Larger x-tick labels
    
    # Add legend with larger font
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

plt.suptitle('Time Series Variation of Representative Monitoring Points', 
             fontsize=18, fontweight='bold')  # Larger title
plt.tight_layout()
plt.savefig('bohai_time_series_small.png', dpi=600, bbox_inches='tight')
plt.show()

# Output basic information
print("\n=== Bohai Sea Environmental Monitoring Data Summary ===")
print(f"Number of monitoring points: {len(location_means)}")
print(f"Data time range: {df['time'].min()} to {df['time'].max()}")
print(f"Longitude range: {df['longitude'].min():.3f}°E - {df['longitude'].max():.3f}°E")
print(f"Latitude range: {df['latitude'].min():.3f}°N - {df['latitude'].max():.3f}°N")

print("\nParameter statistics:")
stats = location_means[['nppv', 'thetao', 'kd', 'chl', 'so', 'spco2']].describe()
print(stats)