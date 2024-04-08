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

from geospatial_tools.geospatial_tools import geotools as gt

# gt contains a model Raster with function of raster and arrays
gtras = gt.Raster() # tools for dealing with rasters
gtgdf = gt.Gdf()    # tools for dealing with dataframe and geodataframe
gtim = gt.imtools() # some useful function related to plotting
gts = gt.Basics()   # a pool of basics tools using toolz library 


```

let's look at the list of functions per module 

```python
dir(gtras)
```

```

'average_maps',
'batch_by_rows',
'categorize_raster',
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
