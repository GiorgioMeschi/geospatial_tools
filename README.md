# geospatial_tools
A set of handy functions for geospatial analysis 

# Installation

```
python -m pip install --no-cache-dir -U git+https://github.com/GiorgioMeschi/geospatial_tools
```
please note you will need to install gdal

# imports 
import 4 main modules 

```python

from geospatial_tools import geotools as gt

# gt contains a model Raster with function of raster and arrays
gtras = gt.Raster() # tools for dealing with rasters
gtgdf = gt.Gdf()    # tools for dealing with dataframe and geodataframe
gtim = gt.Imtools() # some useful function related to plotting
gts = gt.Basics()   # a pool of basics tools using toolz library 


```

let's look at the list of functions per module 

```python
dir(gtras)
```

```

'average_maps',
'average_sliding_windows'
'batch_by_rows',
'categorize_raster',
'contigency_matrix_on_array'
'get_lat_lon_arrays_from_raster',
'max_sliding_windows',
'merge_rasters',
'plot_raster',
'raster_buffer',
'raster_stats_in_polydiss',
'rasterize_gdf_as',
'read_1band',
'remap_raster',
'reproject_raster_as',
'save_raster_as',
'save_raster_multiband',
'write_distance_raster'

```

```python
dir(gtgdf)
```

```
 'aggregate_df',
 'categorize_df',
 'contingency_matrix_df'
 'goupby_and_apply',
 'plot_sequential_gdf',
 'polygonize',
 'zonal_statistics'
```

```python
dir(gtim)
```

```
 'create_distrete_cmap',
 'create_linear_cmap',
 'merge_images',
 'plot_table',
 'ridgeline',
 'style_axis'
```


```python
dir(gts)
```

```
 'add_row_to_df',
 'apply_funcs',
 'eval_freq',
 'filter_dict_keys',
 'filter_dict_values',
 'find_differences',
 'map_dict_keys',
 'map_dict_values',
 'parallel_func_on_input',
 'save_dict_to_json',
 'set_logging
```

# ECAMPLE 1: reproject a raster as reference file and plot it adding 2 shapefiles to the figure
```python

import rasterio as rio
import numpy as np
import geopandas as gpd

from geospatial_tools import geotools as gt

gtras = gt.Raster() # tools for dealing with rasters
gtgdf = gt.Gdf()    # tools for dealing with dataframe and geodataframe


# input 
probabilities_to_plot = f'/share/home/farzad/World_bank/Europe/Bulgaria/Risk/v5/future/SSP585/SSP585_UKESM1-0-LL/2050/allcat/allcat_average_annual_probability.tif'
reference_file = f'/share/home/farzad/World_bank/Europe/Bulgaria/DEM/dem_3035.tif'
# output
prob_reprojected = f'/share/home/farzad/World_bank/Europe/Bulgaria/Risk/v5/future/SSP585/SSP585_UKESM1-0-LL/2050/allcat/temp.tif'

gtras.reproject_raster_as(probabilities_to_plot,
                          prob_reprojected,
                          reference_file = reference_file)

reference = gtras.read_1band(reference_file)
probabilities = gtras.read_1band(prob_reprojected)

# mask and plot

# why to use this function for plottong:
# 1) quick styling of figure 
# 2) possibility to make categorical palette passing only bounds and colors 

# plot the array using categorical palette 
prob_arr_to_plot = np.where(reference == -9999, np.nan, probabilities/100)
gtras.plot_raster(prob_arr_to_plot, shrink_legend = 0.6, dpi = 500, 
                    title = 'Average annual probability of wildfire',
                    # parameters for cateogrical plot
                    array_classes = [0, 0.01, 0.05, 0.08, 1],
                    array_colors = ['blue', 'yellow', 'orange', 'red'],
                    array_names = ['low', 'medium', 'high', 'very high'] )


# this works but I want to plot it passing a rasterio object instead of array, in order to add to the plot some geodataframes

# since no data of original array are not codified correctly, I save the array to plot and open as rasterio object
# save the reprojected files with the parameters I want
gtras.save_raster_as(prob_arr_to_plot, prob_reprojected, reference_file = reference_file, clip_extent = True, dtype = 'float32') # specify dtype to not be integer
probabilities_raster = rio.open(prob_reprojected) # now nodata are correctly codified (as reference file thanks to clip extent paramenter)

# note here i m passing a cmpa, so linear palette is applied
fig, ax = gtras.plot_raster(probabilities_raster, cmap = 'nipy_spectral', shrink_legend = 0.6, dpi = 500, 
                            title = 'Average annual probability of wildfire')


# add to the plot the boundaries and some wildfires
# reading geodataaframes


f1 = f'/share/home/farzad/World_bank/Europe/Bulgaria/Fires/fires.shp'
f2 = f'/share/home/farzad/World_bank/Europe/Bulgaria/Social_vulnerabilities/national_boundaries/jf267dx3808.shp'

working_crs = 'EPSG:3035'

gdf1 = gpd.read_file(f1)
gdf2 = gpd.read_file(f2)

gdf1 = gdf1.to_crs(working_crs)
gdf2 = gdf2.to_crs(working_crs)


# note is cmpa is set to none facecoler is empty, otherwise linear cmpa is used. 
# also in this case quick categorical palette can be define dwith the same 3 input seen for plotting the raster
# add to image 2 dataframes without cmap, only contours
tuples = [(gdf2, 'id_0', {'cmap' : None, 'edgecolor': 'white', 'colorbar' : False, 'linewidth': 0.5, 'linestyle': ':'}), 
          (gdf1, 'area_ha', {'cmap' : None, 'edgecolor': 'red', 'colorbar' : False, 'linewidth': 0.5})]

ax = gtgdf.plot_sequential_gdf(ax, *tuples)

fig

# remove reproj file
os.remove(prob_reprojected)

```

