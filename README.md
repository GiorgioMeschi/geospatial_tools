# geospatial_tools
A set of handy functions for geospatial analysis 

# Installation

```
python -m pip install --no-cache-dir -U git+https://github.com/GiorgioMeschi/geospatial_tools
```
please note you will need to install gdal

# imports 
import main modules 

```python

from geospatial_tools import geotools as gt

# gt contains a model Raster with function of raster and arrays
gtras = gt.Raster() # tools for dealing with rasters
gtgdf = gt.Gdf()    # tools for dealing with dataframe and geodataframe
gtim = gt.Imtools() # some useful function related to plotting
gts = gt.Basics()   # a pool of basics tools using toolz library 

from geospatial_tools import ff_tools
fft = FireTools()

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

```python
dir(gt.Analysis())
```

```
'eval_annual_susc_thresholds',
'hazard_12cl_assesment',
'plot_susc_with_bars',
'plot_training_and_test_dfs'

```


# EXAMPLE 1: reproject a raster as reference file and plot it adding 2 shapefiles to the figure
```python

import rasterio as rio
import numpy as np
import geopandas as gpd

from geospatial_tools import geotools as gt

gtras = gt.Raster() # tools for dealing with rasters
gtgdf = gt.Gdf()    # tools for dealing with dataframe and geodataframe

userp = ''
# input 
probabilities_to_plot = f'{userp}/World_bank/Europe/Bulgaria/Risk/v5/future/SSP585/SSP585_UKESM1-0-LL/2050/allcat/allcat_average_annual_probability.tif'
reference_file = f'{userp}/World_bank/Europe/Bulgaria/DEM/dem_3035.tif'
# output
prob_reprojected = f'{userp}/World_bank/Europe/Bulgaria/Risk/v5/future/SSP585/SSP585_UKESM1-0-LL/2050/allcat/temp.tif'

gtras.reproject_raster_as(probabilities_to_plot,
                          prob_reprojected,
                          reference_file = reference_file)

reference = gtras.read_1band(reference_file)
probabilities = gtras.read_1band(prob_reprojected)

# mask and plot

# why to use this function for plottong:
# 1) quick styling of figure (default clean style) 
# 2) possibility to make categorical palette passing only bounds, color and classes names

# plot the array using categorical palette 
prob_arr_to_plot = np.where(reference == -9999, np.nan, probabilities/100)
gtras.plot_raster(prob_arr_to_plot, shrink_legend = 0.6, dpi = 500, 
                    title = 'Average annual probability of wildfire',
                    # parameters for cateogrical plot
                    array_classes = [0, 0.01, 0.05, 0.08, 1],
                    array_colors = ['blue', 'yellow', 'orange', 'red'],
                    array_names = ['low', 'medium', 'high', 'very high'] )


# this works well but I want to plot it passing a rasterio object instead of array, in order to add to the plot some geodataframes

# since no data of original array are not codified correctly, I save the array to plot and open as rasterio object
# save the reprojected files with the parameters I want
gtras.save_raster_as(prob_arr_to_plot, prob_reprojected, reference_file = reference_file, clip_extent = True, dtype = 'float32') # specify dtype to not be integer
probabilities_raster = rio.open(prob_reprojected) # now nodata are correctly codified (as reference file thanks to clip extent paramenter)

# note here i m passing a cmap, so linear palette is applied
fig, ax = gtras.plot_raster(probabilities_raster, cmap = 'nipy_spectral', shrink_legend = 0.6, dpi = 500, 
                            title = 'Average annual probability of wildfire')


# add to the plot the boundaries and some wildfires
# reading geodataframes
f1 = f'{userp}/World_bank/Europe/Bulgaria/Fires/fires.shp'
f2 = f'{userp}/World_bank/Europe/Bulgaria/Social_vulnerabilities/national_boundaries/jf267dx3808.shp'

working_crs = 'EPSG:3035'

gdf1 = gpd.read_file(f1)
gdf2 = gpd.read_file(f2)

gdf1 = gdf1.to_crs(working_crs)
gdf2 = gdf2.to_crs(working_crs)

# note is cmap is set to none facecoler is empty, otherwise linear cmap is used. 
# also in this case quick categorical palette can be define with the same 3 input seen for plotting the raster

# add to image 2 dataframes without cmap, only contours
tuples = [(gdf2, 'id_0', {'cmap' : None, 'edgecolor': 'white', 'colorbar' : False, 'linewidth': 0.5, 'linestyle': ':'}), 
          (gdf1, 'area_ha', {'cmap' : None, 'edgecolor': 'red', 'colorbar' : False, 'linewidth': 0.5})]

ax = gtgdf.plot_sequential_gdf(ax, *tuples)

fig

# remove reproj file
os.remove(prob_reprojected)

```

# EXAMPLE 2: perform multiple operations on arrays
```python

from geospatial_tools import geotools as gt

import json
import numpy as np

gtras = gt.Raster() # tools for dealing with rasters

# in this example:
# get a wildfire susceptibility with values from 0 to 1
# find thresholds using distribution of values in burned areas to categorize it
# aggregate a vegetation raster in 4 classes of fuel types
# create an hazard layer applying a contingency matrix among susceptibility adn ful type

# inputs
susc_path = f'.tif'
veg_path = f'.tif'
mapping_path = f'.json'
fires = f'.shp'
out_hazard_file = '.tif'

# json contains:
# {
#     "20": "3",
#     "30": "1",
#     "40": "1",
#     "60": "1",
#     "90": "1",
#     "100": "1",
#     "111": "4",
#     "112": "2",
#     "113": "4",
#     "114": "2",
#     "115": "4",
#     "116": "2",
#     "121": "2",
#     "122": "2",
#     "123": "2",
#     "124": "2",
#     "125": "2",
#     "126": "2",
#     "50": "1",
#     "70": "1",
#     "80": "1",
#     "200": "1",
#     "255": "1",
#     "0": "1"
# }

# create a matrix 3 times 4 (susc row entries, ft col entries)
matrix = np.array([[1, 4, 7, 10],
                    [2, 5, 8, 11],
                    [3, 6, 9, 12]])

# open susc, find thresholds, categorize, create fuel type, apply contingency matrix, finally save output
susc = gtras.read_1band(susc_path)
fires_arr = gtras.rasterize_gdf_as(gpd.read_file(fires), susc_path)
susc_clip = np.where(fires_arr == 1, susc, np.nan)
threasholds = np.nanquantile( susc_clip[susc_clip>0], [0.01, 0.2]) 
susc_cl = gtras.categorize_raster(susc, threasholds, nodata = -1)
veg = gtras.read_1band(veg_path)
mapping = json.load(open(mapping_path))
ft = gtras.remap_raster(veg, mapping)
hazard = gtras.contigency_matrix_on_array(susc_cl, ft, matrix, nodatax = 0, nodatay = 0)
gtras.save_raster_as(hazard, out_hazard_file, reference_file = susc_path, dtype = np.int8(), nodata = 0)

# view it quickly continuous palette
for data in [susc_cl, ft, hazard]:
    gtras.plot_raster(data)

# a similar computation can be addressed using the newly gt.Analysis().hazard_12cl_assesment

```
