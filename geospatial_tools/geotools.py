

from dataclasses import dataclass
import rasterio as rio
import logging
import numpy as np
import os
import time
from rasterio import features
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.stats import gaussian_kde
import pandas as pd
import json
from osgeo import gdal
from osgeo.gdal import Polygonize
from osgeo.gdal import ogr
import geopandas as gpd
from rasterio.mask import mask
from rasterio.merge import merge 
from matplotlib import colors
from scipy.ndimage import maximum_filter as maxf2D
import contextily as cx
from PIL import Image
from matplotlib.ticker import FuncFormatter
from rasterio.plot import show as rioshow
import toolz
import types

@dataclass
class Raster:

    def read_1band(self, path: str, band = 1) -> np.array:

        with rio.open(path) as f:
            arr = f.read(band)
            
        return arr

    def save_raster_as(self, array: np.array, output_file: str, reference_file: str, clip_extent = False, **kwargs) -> str:
        '''
        save raster based on reference file, set clip_extent = True to automatically set nodata value as the reference file
        '''
        
        with rio.open(reference_file) as f:
            
            profile = f.profile
            
            profile['compress'] = 'lzw'
            profile['tiled'] =  'True'
    
            profile.update(**kwargs)
                  
            if len(array.shape) == 3:
                array = array[0,:,:]

            if clip_extent == True:
                f_arr= f.read(1)
                noodata = f.nodata
                array = np.where(f_arr == noodata, profile['nodata'], array)
    
            with rio.open(output_file, 'w', **profile) as dst:
                dst.write(array.astype(profile['dtype']), 1)
            
        return output_file

    def save_raster_multiband(self, array: np.array, output_file: str, reference_file: str, band_num: int, **kwargs):
        
        with rio.open(reference_file) as f:
            
            profile = f.profile
            print(f'input profile\n{profile}')
            
            profile['compress'] = 'lzw'
            profile['tiled'] =  'True'
    
            
            profile.update(**kwargs)
            profile.update(count = band_num)
            print(f'output profile\n{profile}')
    
            with rio.open(output_file, 'w', **profile) as dst:
                dst.write(array)    
                
    def reproject_raster_as(self, input_file: str, output_file: str, reference_file: str) -> str:
        '''
        reproj and clip raster based on reference file
        '''

        with rio.open(input_file) as file_i:
            input_crs = file_i.crs
            #bounds = haz.bounds
        with rio.open(reference_file) as ref:
            bounds = ref.bounds
            res = ref.transform[0]
            output_crs = ref.crs
    
        gdal.Warp(output_file, input_file,
                        outputBounds = bounds, xRes=res, yRes=res,
                        srcSRS = input_crs, dstSRS = output_crs, dstNodata = -9999,
                        creationOptions=["COMPRESS=LZW", "PREDICTOR=2", "ZLEVEL=3", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])    
        
        return output_file
        
    def plot_raster(self, raster, cmap = 'seismic', title = '', figsize = (10, 8), dpi = 300, outpath = None,
                    array_classes = [], array_colors = [], array_names = [], shrink_legend = 1, xy = (0.5, 1.1), labelsize = 10,
                    basemap = False, 
                    basemap_params = {'crs' : 'EPSG:4326', 'source' : None, 'alpha' : 0.5, 'zoom' : '4'},
                    add_to_ax: tuple = None) -> tuple:
        
        '''
        plot a raster object with possibility to add basemap and cointinuing to build upon the same ax
        example with discrete palette:
        array_classes = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1],  # all the values including nodata 
        classes_colors = ['#0bd1f700','#0bd1f8', '#1ff238', '#ea8d1b', '#dc1721', '#ff00ff'], # a color for each range 
        classes_names = [ 'no data', 'Very Low', 'Low', 'Medium', 'High', 'Extreme'], # names
        add_to_ax: pass an axis to overlay other object to the same ax. it is a tuple (fig, ax)
        '''
        
        if add_to_ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
        else:
            fig = add_to_ax[0]
            ax = add_to_ax[1]
        
        if len(array_classes) > 0 and len(array_colors) > 0 and len(array_names) > 0:

            cmap = colors.ListedColormap(array_colors)
            norm = colors.BoundaryNorm(array_classes, cmap.N)
                
            # plot the raster
            f = rioshow(raster, ax = ax,
                    cmap = cmap, norm=norm, interpolation='none')
            
            img = f.get_images()[0]                           
                                        
            # trick to shift ticks labels in the center of each color 
            cumulative = np.cumsum(array_classes, dtype = float)
            cumulative[2:] = cumulative[2:] - cumulative[:-2]
            ticks_postions_ = cumulative[2 - 1:]/2
            ticks_postions = []
            ticks_postions.extend(list(ticks_postions_))

            # plot colorbar
            cbar = fig.colorbar(img, boundaries=array_classes, ticks=ticks_postions, shrink = shrink_legend)
            cbar.ax.set_yticklabels(array_names) 
            cbar.ax.tick_params(labelsize = labelsize) 
        else:
            # use imshow so that we have something to map the colorbar to
            image = rioshow(raster, ax = ax,
                        cmap = cmap)
            img = image.get_images()[0]                           
            cbar = fig.colorbar(img, ax=ax,  shrink = shrink_legend) 
            cbar.ax.tick_params(labelsize = labelsize) 

        
        ax.set_xticks([])
        ax.set_yticks([])
        for s in [ "top", 'bottom', "left", 'right']:
            ax.spines[s].set_visible(False)
        
        ax.annotate(title, 
                    xy = xy, xytext = xy, va = 'center', ha = 'center',
                    xycoords  ='axes fraction', fontfamily='sans-serif', fontsize = 12, fontweight='bold')
        
        if basemap:
            if basemap_params['source'] is None:
                cx.add_basemap(ax, crs = basemap_params['crs'], source = cx.providers.OpenStreetMap.Mapnik, alpha = basemap_params['alpha'], 
                                zorder = -1)
            else:
                cx.add_basemap(ax, crs = basemap_params['crs'], source = basemap_params['source'], alpha = basemap_params['alpha'], 
                                zorder = -1, zoom = basemap_params['zoom'])

        if outpath is not None:
            fig.savefig(outpath, dpi = dpi, bbox_inches = 'tight')

        return fig, ax

    def rasterize_gdf_as(self, gdf: gpd.GeoDataFrame, reference_file: str, column = None, all_touched = True) -> np.array:
        '''
        rasterize a shapefile based on a reference raster file
        '''

        with rio.open(reference_file) as f:
            out = f.read(1,   masked = True)
            myshape = out.shape
            mytransform = f.transform #f. ...
        del out    

        out_array = np.zeros(myshape)
        # this is where we create a generator of geom, value pairs to use in rasterizing
        try:
            len(gdf) != 1
            if column is not None:
                shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[column]))
            else:
                shapes = ((geom, 1) for geom in gdf.geometry)
        
        except TypeError: # here it is just 1 point, create an iterable object of 1 point and value
            if column is not None:
                shapes = ( (geom, round(gdf[column])) for geom in [gdf.geometry] )
            else:
                shapes = ( (geom, 1) for geom in [gdf.geometry] )
                print(shapes)
                
        burned = features.rasterize(shapes=shapes, fill=np.NaN, out=out_array, transform=mytransform, all_touched = all_touched)

        return burned
    
    def write_distance_raster(self, raster_file: str, dst_file: str) -> str:
        '''
        calculate the distance af each pixel to the nearest valid one in the raster
        '''

        src_ds = gdal.Open(raster_file)
        srcband=src_ds.GetRasterBand(1)

        drv = gdal.GetDriverByName('GTiff')
        dst_ds = drv.Create( dst_file,
                            src_ds.RasterXSize, src_ds.RasterYSize, 1,
                            gdal.GetDataTypeByName('Float32'))

        dst_ds.SetGeoTransform( src_ds.GetGeoTransform() )
        dst_ds.SetProjection( src_ds.GetProjectionRef() )

        dstband = dst_ds.GetRasterBand(1)
        gdal.ComputeProximity(srcband, dstband, ["DISTUNITS=GEO"])

        return dst_file

    def categorize_raster(self, array: np.ndarray, thresholds: list[int], nodata: float) -> np.array:
        '''
        return an array categorized in calsses on thresholds with out nodata = 0  
        '''
        if np.isnan(np.array(nodata)) == True:
            array = np.where(np.isnan(array)==True, -9E6, array)
            nodata = -9E6
        
        mask = np.where(array == nodata, 0, 1)

        # Convert the raster map into a categorical map based on quantile values
        out_arr = np.digitize(array, thresholds, right=True)
        # make values starting from 1
        out_arr = out_arr + 1 
        out_arr = out_arr.astype(np.int8())

        # mask out no data
        categorized_arr = np.where(mask == 0, 0, out_arr)

        return categorized_arr

    def raster_buffer(self, raster_to_buffer: np.array, pixel_radius: int) -> np.array:

        if len(raster_to_buffer.shape) == 3:
            raster_to_buffer = raster_to_buffer[0,:,:]
        
        kernel = np.ones((2*pixel_radius + 1, 2*pixel_radius + 1))
        
        # clip borders
        raster_no_borders = raster_to_buffer[1:raster_to_buffer.shape[0]-1, 1:raster_to_buffer.shape[1]-1]       
        buffered_mask = ndimage.morphology.binary_dilation(raster_no_borders > 0, structure = kernel)  # , structure=createKernel(1))     
        buffered_img_noBorder = np.where(buffered_mask == True, 1, 0)
        buffered_img = raster_to_buffer.copy()
        
        # add border again with original raster values
        buffered_img[1:buffered_img.shape[0]-1,1:buffered_img.shape[1]-1] = buffered_img_noBorder
        
        return buffered_img

    def remap_raster(self, array: np.array, mapping: dict, nodata = 0) -> np.array:
        '''
        reclassify an array with numpy, make sure all input classes are defined in the mapping table, othewise assertion error will be raised
        passed nodata will be mapped with 0
        '''
        
        input_codes = [ int(i) for i in mapping.keys() ]
        output_codes = [ int(i) for i in mapping.values() ]
  
        # add codification for no data:0
        output_codes.extend([nodata])
        input_codes.extend([0])

        # convert numpy array
        input_codes_arr = np.array(input_codes)
        output_codes_arr = np.array(output_codes)
                
        # check all values in the raster are present in the array of classes
        assert np.isin(array, input_codes_arr).all()
        
        # do the mapping with fancy indexing 
        # find indeces of input codes in input array
        sort_idx = np.argsort(input_codes_arr)
        idx = np.searchsorted(input_codes_arr, array, sorter = sort_idx) # put indeces of input codes in input array
        
        # create an array with indices of new classes linked to indices of input codes, 
        # then in the previous array of indeces put such new classes
        mapped_array = output_codes_arr[sort_idx][idx] 
        
        print(f'List of classes of mapped array: {np.unique(mapped_array)}')
        
        return mapped_array
    
    def max_sliding_windows(self, array: np.ndarray, windows_size: tuple = (3,3)) -> np.array:
    
        # Store shapes of inputs
        N,M = windows_size
        
        if len(array.shape) == 3:
            array = array[0,:,:]
        
        # Use 2D max filter and slice out elements not affected by boundary conditions
        filtered_array = maxf2D(array, size=(M,N))
        
        return filtered_array

    def average_maps(self, paths_to_average: list[str], shp_country = None) -> np.array:
    
        ''' it takes a list of paths to maps clip them WRT shp file and return the average of them'''
        for count, _map in enumerate(paths_to_average):
            raster = rio.open(_map) 

            if shp_country is not None:
                single_map_cliped, _ = rio.mask.mask(raster, shp_country.geometry, crop = True, nodata = np.nan)
            else:
                single_map_cliped = raster.read(1)
            
            if count == 0:
                sum_map = single_map_cliped
            else:
                sum_map += single_map_cliped

        avg_map = sum_map / (count + 1)
        return avg_map   
     
    def raster_stats_in_polydiss(self, array: np.array, gdf: gpd.GeoDataFrame, reference_file: str) -> pd.DataFrame:
        '''
        this types.FunctionType clip the raster of integer classes over dissolved polygons and evaluates the num of pixels per class
        '''
        
        try:
            burned = Raster().rasterize_numerical_feature(gdf, reference_file)
            clipped_img = np.where(burned == 1, array, 0)
            clipped_img = clipped_img.astype(int)
            # burned pixels
            cli = clipped_img[clipped_img > 0]

            # classes and frequency
            _classes, _numbers = np.unique(cli, return_counts = True)         
            classes, frequencies = pd.Series(_classes), pd.Series(_numbers)

            df = pd.DataFrame( columns = ['class', 'num_of_burned_pixels'])
            df['class'] = classes
            df['num_of_burned_pixels'] = frequencies

        except ValueError as e:

            print(e)
            df = pd.DataFrame( columns = ['class', 'num_of_burned_pixels'])

        return df   
    
    def merge_rasters(self, out_path: str, nodata: float, *raster_paths) -> str: # define no data
        
        ras = [rio.open(i) for i in raster_paths]
        out, trans = merge(ras, nodata = nodata)
        Raster().save_raster_as(out, 
                    out_path, raster_paths[0],
                    height = out.shape[1], width = out.shape[2], transform = trans)
        
        return out_path

    def get_lat_lon_arrays_from_raster(self, raster_path: str) -> tuple[np.array, np.array]:
        
        with rio.open(raster_path) as src:

            # Get the geographic coordinate transform (affine transform)
            transform = src.transform
            # Generate arrays of row and column indices
            rows, cols = np.indices((src.height, src.width))
            # Transform pixel coordinates to geographic coordinates
            lon, lat = transform * (cols, rows)

        return lat, lon

    def batch_by_rows(arr_to_batch: np.array, total_batch_num: int, batch_number: int) -> tuple[np.array, int, int]:
        '''
        arr_to_batch is the input array that will be sliced  depending on batch numer 
        tot num of batches is from 1 to x 
        batch number identify the index of the batch, starts from 0
        '''
        rows = arr_to_batch.shape[0]
        interval = int(rows/total_batch_num)
        intervals = [i for i in range(0, rows+1, interval)]
        batch_row_start = intervals[batch_number]
        batch_row_finish = intervals[batch_number + 1]

        # include the missed rows due to rounding operation:
        if total_batch_num - batch_number == 1: # it means last batch
            last_row_wrong = intervals[-1]
            last_row_right = last_row_wrong + (rows - last_row_wrong)
            batch_row_finish = last_row_right
            
        batched_array = arr_to_batch[batch_row_start : batch_row_finish, :]

        return batched_array, batch_row_start, batch_row_finish



@dataclass
class Gdf:

    def aggregate_df(self, df: pd.DataFrame, groupby_col: str, agg_cols: list[str], agg_func: dict) -> pd.DataFrame:

        '''
        index: column of df to group by
        values: columns to aggregate
        aggfunc: dictionary with column names as keys and aggregation types.FunctionTypes as values eg [mean, min, max]
        '''

        aggr_df = df.pivot_table(index = groupby_col, 
                                values = agg_cols, 
                                aggfunc = agg_func)
        
        return aggr_df
    
    def goupby_and_apply(self, df: pd.DataFrame, groupby_col: str, col_to_apply: str, func, out_col = 'output') -> pd.DataFrame:

        '''
        take a dataframe, group and apply types.FunctionType to a column based on the gouped rows, and return the result in a new column
        '''

        df[out_col] = df.groupby(groupby_col)[col_to_apply].transform(func)

        return df
    
    def categorize_df(self, df: pd.DataFrame, col: str, thresholds: list[float], categories: list[str], out_col: str) -> pd.DataFrame:

        df = (df
              .assign(output = lambda x: pd.cut(x[col], bins = thresholds, labels = categories))
              .rename(columns = {'output': out_col})
              )
        
        return df

    def polygonize(self, path_i: str, path_o: str, name = "polygonized") -> str:

        sourceRaster = gdal.Open(path_i)
        band = sourceRaster.GetRasterBand(1)
        
        driver = ogr.GetDriverByName("ESRI Shapefile")
        outDatasource = driver.CreateDataSource(path_o)
        outLayer = outDatasource.CreateLayer(name, srs = None )
        newField = ogr.FieldDefn('CLASSES', ogr.OFTInteger)
        outLayer.CreateField(newField)
        Polygonize(band, None, outLayer, 0, [], callback=None ) # second position is mask..
        outDatasource.Destroy()

        return path_o
    
    def zonal_statistics(self, gdf: gpd.GeoDataFrame, raster_file: str, name_col: str, mode = 'mean', weights: dict = None) -> pd.DataFrame:
        '''
        find statistigs of raster inside polygons. possible value for mode are:
        sum, max, min, q1, q3, weighted_mean, most_frequent
        if mode is weighted_mean pass a list of weightsfor the raster calsses to perform the mean.
        '''
        
        for idx in list(gdf.index):
            # print(idx)
            with rio.open(raster_file) as raster:
                geom = gpd.GeoSeries(gdf.loc[idx].geometry, name = 'geometry')
                try:
                    adm, _= mask(raster, geom, crop = True, nodata = np.NaN)
                except TypeError:
                    adm, _= mask(raster, geom, crop = True, nodata = 0)
                    adm = adm.astype(int)
                    adm = np.where(adm == 0, np.NaN, adm)
                if mode == 'mean':
                    result = np.nanmean(adm)
                elif mode == 'most_frequent':
                    try:
                        result = np.argmax(np.bincount(adm[~np.isnan(adm)].astype(int)))
                    except ValueError: # empty sequence
                        result = np.NaN
                elif mode == 'sum':
                    result = np.nansum(adm)
                elif mode == 'max':
                    result = np.nanmax(adm)
                elif mode == 'min':
                    result = np.nanmin(adm)
                elif mode == 'q1':
                    result = np.nanquantile(adm, 0.25)
                elif mode == 'q3':
                    result = np.nanquantile(adm, 0.75)
                elif mode == 'weighted_mean':
                    classes = weights.keys()
                    weight = list(weights.values())
                    num_pixels = np.nansum(np.where(np.isin(adm, list(classes)), 1, 0)) # count all the pixels excluding the one not in interested classes (no data)
                    percentage_classes = list()
                    for _class in classes:
                        percentage_class = (np.where(adm == _class, 1, 0).sum() / num_pixels) * 100
                        percentage_classes.append(percentage_class)
                    terms = [perc_class * weight for perc_class, weight in zip(percentage_classes, weight)]
                    result = sum(terms)
                    print(f'percentage sum: {percentage_classes}, weight sum: {weight}, result: {result}')
                else:
                    raise ValueError(f'mode {mode} not recognized')
                
                gdf.loc[idx, name_col] = result

        return gdf

    def plot_sequential_gdf(self, ax, *tuples) -> plt.Axes:
        '''       

        each tuple correspond to a gpf to plot adn add to ax, it is structured in the following way:
        [gdf: gpd.GeoDataFrame, colname: str, settings: dict)] where settigns is a dictionary with the following default data: 
        
        cmap = 'viridis', figsize = (10,8), edgecolor = 'black', title = '', xy = (0.5, 1.1),
        array_classes = [], array_colors = [], array_names = [], shrink_legend = None, dpi = 200, 
        vmax = None, outpath = None, colorbar = True
        
        example with discrete palette:
        array_classes = [-1,-0.8, -0.2, 0, 0.5, 1.5, 8],  # all the values including nodata 
        array_colors = ['#0bd1f700','#0bd1f8', '#1ff238', '#ea8d1b', '#dc1721', '#ff00ff'], # a color for each range 
        array_names = [ 'no data','Very Low', 'Low', 'Medium', 'High', 'Extreme'], # names
        '''

        # define default settings stored in third position of tuple
        default_settings = dict(cmap = 'viridis', figsize = (10,8), edgecolor = 'black', title = '', xy = (0.5, 1.1),
        array_classes = [], array_colors = [], array_names = [], shrink_legend = None, dpi = 200, 
        vmax = None, outpath = None, colorbar = True)

        for mytuple in tuples:
            
            default = default_settings.copy()

            # initialize the tuple
            if mytuple[2] is not None:
                default.update(mytuple[2])

            # extract all the elements for plotting
            gdf, colname, _ = mytuple
            cmap, figsize, edgecolor, title, xy, array_classes, array_colors, array_names, shrink_legend, dpi, vmax, outpath, colorbar = default.values()

            if len(array_classes) > 0 and len(array_colors) > 0 and len(array_names) > 0:

                cmap = colors.ListedColormap(array_colors)
                norm = colors.BoundaryNorm(array_classes, cmap.N)

                # trick to shift ticks labels in the center of each color 
                cumulative = np.cumsum(array_classes, dtype = float)
                cumulative[2:] = cumulative[2:] - cumulative[:-2]
                ticks_postions_ = cumulative[2 - 1:]/2
                ticks_postions = []
                ticks_postions.extend(list(ticks_postions_))

                if shrink_legend is None:
                    legend_kwds = {'ticks': ticks_postions,
                                    'boundaries': array_classes}
                else:
                    legend_kwds = {'ticks': ticks_postions, 'shrink': shrink_legend, 
                                    'boundaries': array_classes}
                # plot the shapefile
                gdf.plot(colname, cmap = cmap, norm = norm, edgecolor = edgecolor,
                            figsize = figsize, legend = colorbar,
                            legend_kwds = legend_kwds, 
                            ax = ax,
                            #legend_labels = classes_names
                            )  
                if colorbar:
                    cbar = ax.get_figure().get_axes()[1]
                    cbar.set_yticklabels(array_names)

            else: 
                
                # automatically infer facecolor if cmap is set to None
                try:
                    noerror = mytuple[2]['cmap']
                    facecolor = 'none' if noerror == None else 'none'
                except TypeError:
                    facecolor = None

                if vmax is not None: 
                    ticks_postions = np.linspace(round(gdf[colname].min()), vmax, 6)
                    legend_kwds = {'shrink': shrink_legend, 'ticks': ticks_postions} if shrink_legend is not None else {'ticks': ticks_postions}  
                    gdf.plot(colname, cmap = cmap, edgecolor = edgecolor, figsize = figsize, legend = colorbar,
                                legend_kwds = legend_kwds, vmax = vmax,
                                ax = ax, facecolor = facecolor)   

                    if colorbar:           
                        cbar = ax.get_figure().get_axes()[1]
                        labels = [str(round(i,2)) if i < 1 else str(round(i)) for i in ticks_postions]
                        labels[-1] = labels[-1] + f' to {round(gdf[colname].max())}'
                        cbar.set_yticklabels(labels)
                else:
                    legend_kwds = {'shrink': shrink_legend} if shrink_legend is not None else None

                    gdf.plot(colname, cmap = cmap, edgecolor = edgecolor, figsize = figsize, legend = colorbar,
                                legend_kwds = legend_kwds, ax = ax, facecolor = facecolor)  


            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            # plot text in center top of image
            ax.annotate(title, 
                        xy = xy, xytext = xy, va = 'center', ha = 'center',
                        xycoords  ='axes fraction', fontfamily='sans-serif', fontsize = 12, fontweight='bold')

            if outpath is not None:
                plt.savefig(outpath, dpi = dpi, bbox_inches = 'tight')
        
        return ax
        
    def contingency_matrix_df(self, df: pd.DataFrame | gpd.GeoDataFrame, 
                            colname_rows: str, colname_cols: str, colname_out: str,
                            contincency_values: np.array, contingency_cols: list, contingency_rows: list
                            ) -> pd.DataFrame | gpd.GeoDataFrame:
        '''
        df: dataframe or geodataframe
        colname_rows: name of column as rows of continency matrix
        colname_cols: name of column as cols of continency matrix
        colname_out: name of column after apply contingency matrix
        contincency_values: matrix values as array
        contingency_cols: column names of matrix
        contingency_rows: row names of matrix
        return a dataframe with the new columns after applying the matrix 
        if value in df is not in the matrix, row output will be set as nan
        '''


        contingency_df = pd.DataFrame(contincency_values, columns = contingency_cols, index = contingency_rows)

        def contingency(row, contingency_df):
            try:
                return contingency_df.loc[row[colname_rows], row[colname_cols]]
            except KeyError:
                return np.nan


        df[colname_out] = df.apply(lambda x: contingency(x, contingency_df), axis = 1)

        return df


@dataclass
class Basics:

    def eval_freq(self, x: list) -> dict:
        return toolz.frequencies(x)
    
    def find_differences(self, list1: list, list2: list):
        return list(toolz.diff(list1, list2))
    
    def apply_funcs(self, x, *funcs):
        return toolz.pipe(x, *funcs)
    
    def parallel_func_on_input(self, x, funcs: list[types.FunctionType]):
        return toolz.juxt(funcs)(x)
    
    def map_dict_values(self, x: dict, func: types.FunctionType):
        return toolz.valmap(func, x)
    
    def map_dict_keys(self, x: dict, func: types.FunctionType):
        return toolz.keymap(func, x)
    
    def filter_dict_values(self, x: dict, func: types.FunctionType):
        return toolz.valfilter(func, x)
    
    def filter_dict_keys(self, x: dict, func: types.FunctionType):
        return toolz.keyfilter(func, x)
    
    def add_row_to_df(self, df, dict_of_values: dict):
        '''
        add a row to a dataframe matching the columns
        dict of values: dictionary associating name of columns with values to add in the row of df
        '''

        if df.index.name == None:
            name_index = 'index'
        else:
            name_index = df.index.name

        columns = list(dict_of_values.keys())
        values = list(dict_of_values.values())
        row_to_append = np.array(values).reshape(1, len(columns))                    
        df_temp = pd.DataFrame(row_to_append, columns = columns)
        df = pd.concat([df, df_temp], axis = 0).reset_index().drop(name_index, axis = 1)

        return df
    
    def set_logging(self, working_dir: str, level = logging.INFO):

        log_path = os.path.join(working_dir, 'loggings')
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        logging_date = time.time()
        logging_date = time.strftime("%Y-%m-%d", time.localtime(logging_date))

        p =  os.path.join(log_path, f'log_{logging_date}.log')
        print(f'find the logging here \n{p}')

        logging.basicConfig(format = '[%(asctime)s] %(filename)s: {%(lineno)d} %(levelname)s - %(message)s',
                            datefmt ='%H:%M:%S',
                            filename = p,
                            level = level)  
        
        logging.info(f'\n\n NEW SESSION \n\n')

    def save_dict_to_json(self, dictionary: dict, outpath: str):

        json_obj = json.dumps(dictionary, indent=4)
        with open(outpath, "w") as outfile:
            outfile.write(json_obj)


@dataclass
class imtools:

    def plot_table(self, df: pd.DataFrame, title: str, outpath: str = None, 
                   fontsize = 11, figsize = (10,8), dpi = 200, cellColours: np.array = None,
                   xlabel = None, ylabel = None) -> tuple:
        
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)

        table = ax.table(np.array(df), loc = 'center', cellLoc = 'center', rowLoc = 'center',
                        bbox = [0, 0.02, 1, 1], colLabels = ylabel, rowLabels = xlabel, 
                        cellColours = cellColours)

        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1.5, 1.5)

        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        ax.set_xticks([])   
        ax.set_yticks([])

        fig.text(0.5, 0.92, f'{title}', ha = 'center', fontsize = 16, fontweight = 'bold')

        if outpath is not None:
            fig.savefig(outpath, dpi = dpi, bbox_inches = 'tight')
            
        return fig, ax

    def style_axis(self, ax, **kwards) -> plt.Axes:
        '''
                _kwards = dict(
                legend_bbox = (0.95, 1), 
                legend_font= 8, 
                xlim1= 0, 
                xlim2= 100, 
                ylim1= 0, 
                ylim2= 100 , 
                xticks= np.arange(0,100,10),
                xticklabels= np.arange(0,100,10), 
                yticks= np.arange(0,100,10), 
                yticklabels= np.arange(0,100,10),
                xlabel= 'XLABEL', 
                ylabel= 'YLABEL', 
                label_font= 10,
                commasep_y= False,
                del_spines= ['bottom', 'top', 'left', 'right'], 
                grid= False, 
                colorgrid= 'gray', 
                linestylegrid= '--', 
                linewidthgrid= 0.5, 
                title= 'TITLE', 
                title_annot= (0.5, 1.1), 
                title_font= 15, 
                message= None, 
                annot_arrow= None, 
                annot_text= None, 
                message_font= None

        )

        '''


        _kwards = dict(
                legend_bbox = (0.95, 1), 
                legend_font= 8, 
                xlim1= 0, 
                xlim2= 100, 
                ylim1= 0, 
                ylim2= 100 , 
                xticks= np.arange(0,100,10),
                xticklabels= np.arange(0,100,10), 
                yticks= np.arange(0,100,10), 
                yticklabels= np.arange(0,100,10),
                xlabel= 'XLABEL', 
                ylabel= 'YLABEL', 
                label_font= 10,
                commasep_y= False,
                del_spines= ['bottom', 'top', 'left', 'right'], 
                grid= False, 
                colorgrid= 'gray', 
                linestylegrid= '--', 
                linewidthgrid= 0.5, 
                title= 'TITLE', 
                title_annot= (0.5, 1.1), 
                title_font= 15, 
                message= None, 
                annot_arrow= None, 
                annot_text= None, 
                message_font= None

        )

        _kwards.update(**kwards)

        if _kwards['legend_bbox'] is not None:
            ax.legend(loc = 'upper right', fontsize = _kwards['legend_font'], bbox_to_anchor = _kwards['legend_bbox'], frameon = False)
        ax.set_xlim(_kwards['xlim1'], _kwards['xlim2'])
        ax.set_ylim(_kwards['ylim1'], _kwards['ylim2'])
        ax.set_xticks(_kwards['xticks'])
        ax.set_xticklabels(_kwards['xticklabels'])
        ax.set_yticks(_kwards['yticks'])
        ax.set_yticklabels(_kwards['yticklabels'])
        ax.set_xlabel(_kwards['xlabel'], fontfamily='sans-serif', fontsize = _kwards['label_font'])
        ax.set_ylabel(_kwards['ylabel'], fontfamily='sans-serif', fontsize = _kwards['label_font'])

        if _kwards['commasep_y']:
            ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        
        for s in _kwards['del_spines']:
            ax.spines[s].set_visible(False)
        
        if _kwards['grid']:
            ax.grid(color = _kwards['colorgrid'], linestyle = _kwards['linestylegrid'], linewidth = _kwards['linewidthgrid'])

        ax.annotate(_kwards['title'],
                    xy = _kwards['title_annot'], xytext = _kwards['title_annot'], va = 'center', ha = 'center',
                    xycoords  ='axes fraction', fontfamily='sans-serif', fontsize = _kwards['title_font'], fontweight='bold')

        if _kwards['message'] is not None:
            ax.annotate(_kwards['message'], 
                    xy = _kwards['annot_arrow'], xytext = _kwards['annot_text'], 
                    arrowprops = None if _kwards['annot_arrow'] == None else dict(facecolor = 'black',  arrowstyle = '<-'),
                    xycoords  ='axes fraction', fontfamily='sans-serif', 
                    fontsize = _kwards['message_font'])
            
        return ax
    
    def create_distrete_cmap(self, all_bounds, color_list = ['#0bd1f8', '#1ff238', '#ea8d1b', '#dc1721', '#ff00ff']) -> tuple:
        cmap = colors.ListedColormap(color_list)
        norm = colors.BoundaryNorm(all_bounds, cmap.N)

        return cmap, norm

    def create_linear_cmap(self, min_value: int, max_value: int, color_list = ["lavenderblush","deeppink","purple"]):

        cvals = np.linspace(min_value, max_value, len(color_list))
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list( zip( map(norm, cvals), color_list ))
        my_custom_cmap = colors.LinearSegmentedColormap.from_list("custom palette", tuples)

        return my_custom_cmap

    def merge_images(self, files: list[str], nrow: int, ncol: int):

        images = list()
        for filepath in files:
            images.append(Image.open(filepath))

        # resize them to the same size
        images = [i.resize(images[0].size) for i in images]

        # get the max heigh and width
        new_width = images[0].width * ncol
        new_height = images[0].height * nrow

        # create new img, I put 3 up and 3 down
        new_image = Image.new('RGB', (new_width, new_height))

        # update the y position for pasting img for each row
        i = 0 # index for definig the correct images in the list
        y_offset = 0
        for row in range(nrow):
            x_offset = 0
            for im in images[i : ncol+i]:
                new_image.paste(im, (x_offset, y_offset))
                x_offset += im.size[0]
            y_offset += im.size[1]
            i += ncol

        return new_image

    def ridgeline(data: list[list], overlap = 0.5, fill = True, labels = None, n_points = 1000) -> tuple:
        """
        Creates a standard ridgeline plot.

        data, list of lists, values for each x axis.
        overlap, overlap between distributions. 1 max overlap, 0 no overlap.
        fill, matplotlib color to fill the distributions.
        n_points, number of points to evaluate each distribution function.
        labels, values to place on the y axis to describe the distributions.
        return fig and ax objects.
        """

        fig, ax = plt.subplots(dpi = 200)
        if overlap > 1 or overlap < 0:
            raise ValueError('overlap must be in [0 1]')
        xx = np.linspace(np.min(np.concatenate(data)),
                        np.max(np.concatenate(data)), n_points)
        curves = []
        ys = []
        for i, d in enumerate(data):
            pdf = gaussian_kde(d)
            y = i*(1.0 - overlap)
            ys.append(y)
            curve = pdf(xx)
            if fill:
                ax.fill_between(xx, np.ones(n_points)*y, 
                                curve+y, zorder=len(data)-i+1, color=fill)
            ax.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
        if labels:
            ax.set_yticks(ys, labels)
        
        return fig, ax





#%%

# f1 = f'/share/home/farzad/World_bank/Europe/Croatia/Fires/fires.shp'
# f2 = f'/share/home/farzad/World_bank/Europe/Croatia/Social_vulnerabilities/national_boundaries/level1.shp'

# gdf1 = gpd.read_file(f1)
# gdf2 = gpd.read_file(f2)

# gdf1 = gdf1.to_crs(gdf2.crs)

# fig,ax = plt.subplots(dpi = 200)

# tuples = [(gdf2, 'id_0', {'cmap' : None, 'colorbar' : False}), 
#           (gdf1, 'area_ha', {'cmap' : 'inferno', 'edgecolor' : None})]

# ax = plot_sequential_gdf('', ax, *tuples)




# %%
