
#imports
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio.mask import mask as riomask
import json
from dataclasses import dataclass

from geospatial_tools import geotools as gt



@dataclass
class FireTools:

    '''
    Some useful functions to apply on  wildfire susceptibility and hazard processing
    '''

    def plot_susc_with_bars(self, fires_file: str | gpd.GeoDataFrame | None, fires_col: str, crs:str, susc_path: str, 
                            xboxmin_hist: float, yboxmin_hist: float, xboxmin_pie: float, yboxmin_pie: float,
                            threshold1: float, threshold2: float, out_folder: str, year: int = 'Present', month=None,
                            season = False, total_ba_period = None, susc_nodata = -1, pixel_to_ha_factor = 1,
                            allow_hist = True, allow_pie = True, allow_fires = True,
                            normalize_over_y_axis: int | None = 20, limit_barperc_to_show: int = 0) -> tuple:
        
        '''
        Plot susceptibility map categorized with fires and histogram and pie showing statistics.
        fires analysis can be excluded if fires_file is None
        fires plot, histogram and pie can be removed tuning allow_* parameters
        xboxmin and yboxmin define the position of histogram and pie in the figure
        threshold1 and 2 are the thresholds for categorization of suscpetibility 
        total_ba_period: total burned area in a period at choice, 
        the percentage of ba per each histogram bar will be computed w.r.t this number as additional info
        pixel_to_ha_factor: conversion from pixel resolution to hectar, if res is 100m factor is 1
        if allow hist is true, normalize_over_y_axis define the max bar hight wrt total_ba_period \n
        while limit_barperc_to_show is the limit percentage value to show inside the bar
        '''

        gtras = gt.Raster()

        def histogram(stats, ax, total_ba_period):
            '''
            create bar plot with the stats of burned area per susceptibility class
            '''

            b = ax.bar(stats['class'], stats.num_of_burned_pixels, width = 0.4, color = 'brown') 
            # eliminate axes visiblity 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks(stats['class'])
            alllabels = {1: 'Low', 2:'Medium', 3: 'High'}
            labels = [alllabels[i] for i in stats['class']]
            ax.set_xticklabels(labels)
            ax.tick_params(axis='x', labelsize = 9)
            ax.annotate('Burned area per susceptibility class [ha]', xytext =  (-0.05, 1.2),
                        xy = (-0.05, 1.2), fontsize = 7, fontweight = 'bold', xycoords = 'axes fraction')

            for v, y in zip(stats.num_of_burned_pixels, stats['class']):
                c = 1.05
                v = int(v)
                ax.annotate(f'{v:,}', xy = (y, v*c), xytext = (y, v*c), 
                            ha = 'center', va = 'bottom', fontsize = 8, fontweight = 'bold',
                            color = 'black', zorder = 15)
                
                if total_ba_period is not None:
                    perc = (v/total_ba_period) * 100
                    if perc > limit_barperc_to_show:
                        ax.annotate(f'{perc:.0f}%', xy = (y, v/2), xytext = (y, v/2), 
                                    ha = 'center', va = 'bottom', fontsize = 6, fontweight = 'bold',
                                    color = 'white', zorder = 15)

            ax.set_xlim(0.5, 3.5)
            if total_ba_period is not None:
                if normalize_over_y_axis is not None:
                    ax.set_ylim(0, normalize_over_y_axis * total_ba_period /100)

            return ax
            
        def pie(ax, susc_arr, susc_nodata, pixel_to_ha_factor):
            '''
            create pie plot with the extent of susc classes
            '''
            # count the number of pixels per class
            vals, counts = np.unique(susc_arr[susc_arr != susc_nodata], return_counts = True)
            counts = counts * pixel_to_ha_factor
            percentage = counts/counts.sum() * 100
            ax.pie(counts, autopct='%1.0f%%', startangle=90, 
                    colors = ['green', 'yellow', 'red'], 
                    textprops={'color':"black", 'size': 9, 'weight':'bold'})

            spines = ['top', 'right', 'left', 'bottom']
            for s in spines:
                ax.spines[s].set_visible(False)

            ax.set_yticks([])
            ax.set_xticks([])

            return ax
        
        if fires_file is not None:
            # if file is string, read it
            if isinstance(fires_file, str): 
                fires = gpd.read_file(fires_file)
                # filter only from: 2008 
                fires[fires_col] = pd.to_datetime(fires[fires_col])
            else:
                fires = fires_file
            
            fires = fires.to_crs(crs)

            annualfire = fires[(fires[fires_col].dt.year == year)] if isinstance(year, int) else fires.copy()
            if season == True:
                months = list(range(4,11)) if month == 1 else list(range(1, 4)) + list(range(11, 13)) 
                annualfire = annualfire[(annualfire[fires_col].dt.month.isin(months))] 
            else:
                annualfire = annualfire[(annualfire[fires_col].dt.month == month)] if month is not None else annualfire
        
        annualsusc = rio.open(susc_path)

        fig, ax = plt.subplots(figsize=[12,10], dpi = 250)

        allow_fires = False if fires_file is None else allow_fires
        if allow_fires:
            if len(annualfire) != 0:
                annualfire.plot(ax = ax, edgecolor = 'black', linewidth = 0.9, facecolor = 'none')

        month_label = "" if month is None else 'Summer' if month == 1 and season == True else 'Winter' if month == 2 and season == True else month 
        month_labels_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        if isinstance(month_label, int):
            month_label = month_labels_dict[month_label]
        fig, ax = gtras.plot_raster(annualsusc,
                                    add_to_ax = (fig, ax), 
                                    # define the settings for discrete plotting
                                    array_classes = [-2, -0.5, threshold1, threshold2, 1],
                                    array_colors = ['#0bd1f700','green', 'yellow', 'red'],
                                    array_names = [ 'no data', 'Low', 'Medium', 'High'],
                                    title = f'Susceptibility {year} {month_label}',
                                    shrink_legend=0.4,
                                )
        allow_hist = False if fires_file is None else allow_hist
        if allow_hist:
            if len(annualfire):

                ax1 = fig.add_axes([xboxmin_hist, yboxmin_hist, 0.15, 0.13])  # left, bottom, width, height
                susc_class = gtras.categorize_raster(annualsusc.read(1), 
                                        thresholds = [threshold1, threshold2],
                                        nodata = susc_nodata)
            
                stats = gtras.raster_stats_in_polydiss(susc_class, annualfire, reference_file = susc_path)
                stats['num_of_burned_pixels'] = stats.num_of_burned_pixels * pixel_to_ha_factor
                # insert the plot in the same figure and separate axes
                ax1 = histogram(stats, ax1, total_ba_period)
        
        if allow_pie:
            try:
                susc_class.shape
            except:
                susc_class = gtras.categorize_raster(annualsusc.read(1), 
                                                thresholds = [threshold1, threshold2],
                                                nodata = susc_nodata)

            #plot pie chart with classes extent
            ax2 = fig.add_axes([xboxmin_pie, yboxmin_pie, 0.15, 0.15])
            ax2 = pie(ax2, susc_class, 0, pixel_to_ha_factor)

        os.makedirs(out_folder, exist_ok = True)
        fig.savefig(f'/{out_folder}/susc_plot_{year}{month}.png', dpi = 200, bbox_inches = 'tight')

        return fig, ax

    def add_avg_trend_to_fig(self, fig, years: list[int], avg_arrs: np.array, year:int, 
                            xboxmin = 0.68, yboxmin = 0.08):

        '''
        add optional average susceptibility trend over the time window considered

        fig: input matplotlib figure
        years: list of years
        avg_arrs: array of average susceptibility values 
        year: selected year
        xboxmin: x position of the box
        yboxmin: y position of the box

        return the new figure
        '''

        # plot trend of average susc adding as new axis
        ax3 = fig.add_axes([xboxmin, yboxmin, 0.12, 0.11])
        ax3.plot(years, avg_arrs, color = 'lightblue')
        title = 'Trend of average susceptibility'
        ax3.annotate(title, xytext =  (-0.1, 1.2),
                    xy = (-0.05, 1.2), fontsize = 7, fontweight = 'bold', xycoords = 'axes fraction')

        # off right and top 
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        # x ticks off
        ax3.set_xticks(years)

        # change color of the edge of the marker
        ax3.plot(year, avg_arrs[years.index(year)], marker = '*', color = 'yellow', markersize = 10, markeredgecolor = 'black')
        # add ticks label only for the first, selected and last years
        ax3.set_xticklabels(years, rotation = 45, fontsize = 7)
        # make some years labels disappear
        for i, label in enumerate(ax3.get_xticklabels()):
            if i not in [0, years.index(year), len(years)-1]:
                label.set_visible(False)

        return fig 

    def hazard_12cl_assesment(self, susc_path: str, thresholds: list, veg_path: str, mapping_path: str, out_hazard_file: str) -> tuple:
        '''
        susc path is the susceptibility file, contineous values, no data -1
        threasholds are the values to categorize the susceptibility (3classes)
        veg_path is the input vegetation file
        mapping_path is the json file with the mapping of vegetation classes (input veg: output FT class from 1 to 4)
        where FT class are 1: grasslands, 2: broadleaves, 3: shrubs, 4: conifers.
        out_hazard_file is the output hazard file
       
        Return: wildfire hazard, susc classes and fuel type array
        '''

        gtras = gt.Raster()
        matrix = np.array([ [1, 4, 7, 10],
                            [2, 5, 8, 11],
                            [3, 6, 9, 12]])
                
        susc = gtras.read_1band(susc_path)
        susc_cl = gtras.categorize_raster(susc, thresholds, nodata = -1)
        veg = gtras.read_1band(veg_path)
        mapping = json.load(open(mapping_path))
        ft = gtras.remap_raster(veg, mapping)
        hazard = gtras.contigency_matrix_on_array(susc_cl, ft, matrix, nodatax = 0, nodatay = 0)
        gtras.save_raster_as(hazard, out_hazard_file, reference_file = susc_path, dtype = np.int8(), nodata = 0)
        
        return hazard, susc_cl, ft
    
    def eval_annual_susc_thresholds(self, countries: list[str], years, 
                                folder_before_country: str, folder_after_country: str,
                                fires_paths: str, name_susc_without_year: str = 'Annual_susc_', 
                                year_fires_colname: str = 'finaldate', crs = 'EPSG:3035', year_in_name = True,
                                allow_plot = True):

        '''
        compute trasholds on annual wildfire susceptibilities: 
        # compute annual 90% and 1% values on the burnead area susceptibility values distribution
        # tr 1: min of 1% values of the years above average burned area 
        # tr 2: min of 90% values of the years above average burned area  

        annual susceptibilities have this structure path:
        f'{folder_before_country}/{country}/{folder_after_country}/{name_susc_without_year}{year}.tif'
        if year_in_name is Fakse the year is excluded in the filename
        fire_path is this one:
        f'{folder_before_country}/{country}/{fires_paths}'
        average susceptibility has this path:
        f'{folder_before_country}/{country}/{path_after_country_avg_susc}'

        
        return threasholds dict, high_vals_years list, low_vals_years list, ba_list
        '''

        gtras = gt.Raster()

        high_vals_years = list()
        low_vals_years = list()
        ba_list = list()
        for year in years:
            vals_years = list()
            all_fires = []
            for country in countries:
                if year_in_name == True:
                    country_paths_to_check = f'{folder_before_country}/{country}/{folder_after_country}/{name_susc_without_year}{year}.tif'
                else:
                    country_paths_to_check = f'{folder_before_country}/{country}/{folder_after_country}/{name_susc_without_year}.tif'
                path = country_paths_to_check
                fire_p = f'{folder_before_country}/{country}/{fires_paths}'
                fires = gpd.read_file(fire_p)
                fires[year_fires_colname] = pd.to_datetime(fires[year_fires_colname])
                if len(str(year)) == 4:
                    fires = fires[(fires[year_fires_colname].dt.year == year)]
                else:
                    year, month = year.split('_')
                    fires = fires[(fires[year_fires_colname].dt.year == int(year))]
                    fires = fires[(fires[year_fires_colname].dt.month == int(month))]
                fires = fires.to_crs(crs)
                if len(fires) != 0:
                    susc = rio.open(path)
                    susc_clip, _ = riomask(susc, fires.geometry, nodata = -1)
                    vals = list(susc_clip[susc_clip != -1])
                    vals_years.append(vals)
                    all_fires.append(fires)

            if len(all_fires) != 0:
                all_fires_df = pd.concat(all_fires)
                all_fires_df= all_fires_df.to_crs(crs)
                all_fires_df['ba'] = all_fires_df.area / 10000 
                total_ba = all_fires_df['ba'].sum()
                ba_list.append(total_ba)

                # flat list
                vals_years = [item for sublist in vals_years for item in sublist]
                quntiles = np.quantile(vals_years, [0.01, 0.1])
                high_vals_years.append(quntiles[1])
                low_vals_years.append(quntiles[0])

                if allow_plot:
                    # plot vals_year with 2 vertical bars of quantiles:
                    fig, ax = plt.subplots(dpi = 200)
                    ax.hist(vals_years, bins = 50, color = 'blue')
                    ax.axvline(quntiles[0], color='green', linestyle='dashed', linewidth=1)
                    ax.axvline(quntiles[1], color='red', linestyle='dashed', linewidth=1)
                    ax.set_title(f'Distribution of Susceptibility values in annual burned areas {year}', fontweight = 'bold', fontsize = 10)
            else:
                high_vals_years.append(0)
                low_vals_years.append(0)
                ba_list.append(0)

        avg_ba = np.mean(ba_list)
        mask_over_treashold = [1 if ba > avg_ba else 0 for ba in ba_list]

        # select values from high and  low vals
        mask = np.array(mask_over_treashold)
        high_vals_years = np.array(high_vals_years)
        low_vals_years = np.array(low_vals_years)

        high_val_over_tr =  high_vals_years[mask == 1]
        low_val_over_tr =  low_vals_years[mask == 1]
        # low_val_under_tr = low_vals_years[mask == 0]
        # high_val_under_tr = high_vals_years[mask == 0]

        # case 1
        lv2 = min(high_val_over_tr)
        lv1 = min(low_val_over_tr)

        # case 2
        # min(high_val_over_tr)
        # max(low_val_under_tr)

        thresholds = {'lv1' : lv1, 'lv2' : lv2}

        return thresholds, high_vals_years, low_vals_years, ba_list
    
    def plot_training_and_test_dfs(self, X_training_p: str, X_test_p: str, Y_training_p:str, Y_test_p:str, cols: list, outfolderpath: str):

        # open x
        X_train = np.load(X_training_p)
        X_test = np.load(X_test_p)
        X_train = pd.DataFrame(X_train, columns = cols)
        X_test = pd.DataFrame(X_test, columns = cols)

        # open Y
        Y_train = np.load(Y_training_p)
        Y_test = np.load(Y_test_p)

        # convert to dataframe
        Y_train = pd.DataFrame(Y_train, columns = ['label'])
        Y_test = pd.DataFrame(Y_test, columns = ['label'])

        # filter X when Y is 1
        X_train = X_train[Y_train['label'] == 1]
        X_test = X_test[Y_test['label'] == 1]

        # plot coords, lat and lon
        fig, ax = plt.subplots(figsize=[12,10], dpi = 250)
        X_train.plot.scatter(x = 'lon', y = 'lat', ax = ax, color = 'blue', s = 0.1, label = 'Training', zorder = 10)
        X_test.plot.scatter(x = 'lon', y = 'lat', ax = ax, color = 'red', s = 0.1, label = 'Test', zorder = 1)
        ax.legend()
        ax.set_title('Training and Test data')
        # remove axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # convert to geo df the 2 dataframes
        X_train = gpd.GeoDataFrame(X_train, geometry = gpd.points_from_xy(X_train.lon, X_train.lat))
        X_test = gpd.GeoDataFrame(X_test, geometry = gpd.points_from_xy(X_test.lon, X_test.lat))

        # save the dataframes
        os.makedirs(outfolderpath, exist_ok = True)
        X_train.to_file(f'{outfolderpath}/X_train_coords.shp')
        X_test.to_file(f'{outfolderpath}/X_test_coords.shp')

    def plot_haz_with_bars(self, fires_file: str | gpd.GeoDataFrame | None, fires_col: str, crs:str, hazard_path: str, 
                            xboxmin_hist: float, yboxmin_hist: float, xboxmin_pie: float, yboxmin_pie: float,
                            out_folder: str, year: int = 'Present', month=None,
                            season = False, haz_nodata = 0, pixel_to_ha_factor = 1,
                            allow_hist = True, allow_pie = True, allow_fires = True) -> plt.figure:
        
        '''
        Plot hazard map categorized with fires and histogram and pie showing statistics.
        fires analysis can be excluded if fires_file is None
        fires plot, histogram and pie can be removed tuning allow_* parameters
        xboxmin and yboxmin define the position of histogram and pie in the figure
        threshold1 and 2 are the thresholds for categorization of suscpetibility 
        total_ba_period: total burned area in a period at choice, 
        the percentage of ba per each histogram bar will be computed w.r.t this number as additional info
        pixel_to_ha_factor: conversion from pixel resolution to hectar, if res is 100m factor is 1
        '''

        gtras = gt.Raster()

        def histogram(stats, ax):
            '''
            create bar plot with the stats of burned area per susceptibility class
            '''

            b = ax.bar(stats['class'], stats.percentage, width = 0.4, color = 'brown') #0.4
            # eliminate axes visiblity 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # ax.set_yticks([])
            ax.set_xticks(stats['class'])
            alllabels = list(range(1,13))
            labels = [i for i in stats['class']]
            ax.set_xticklabels(labels)
            ax.tick_params(axis='x', labelsize = 9)
            ax.tick_params(axis='y', labelsize = 8)
            ax.annotate('Burned area per Hazard class [% w.r.t. class extent]', xytext =  (-0.05, 1.2),
                        xy = (-0.05, 1.2), fontsize = 7, fontweight = 'bold', xycoords = 'axes fraction')

                
            ax.set_xlim(0.5, 12.5)

            return ax
            
        def pie(ax, haz_arr, haz_nodata, pixel_to_ha_factor):
            '''
            create pie plot with the extent of susc classes
            '''
            # count the number of pixels per class
            labels, counts = np.unique(haz_arr[haz_arr != haz_nodata], return_counts = True)
            counts = counts * pixel_to_ha_factor
            percentage = counts/counts.sum() * 100
            percentage_format = [f'{p:.0f}%' for p in percentage]
            all_colors = {"1": "#99ff99", "2": "#00ff00", "3": "#006600", "4": "#ffff99", "5": "#ffff00", "6": "#cc9900", "7": "#cc99ff",
                        "8": "#9933cc", "9": "#660099", "10": "#f55b5b", "11": "#ff0000", "12": "#990000"}
            colors = [all_colors[str(label)] for label in labels]
            explode_funz = lambda x: 0.1 if x >10 else 0.22
            explode = [explode_funz(p) for p in percentage]
            ax.pie(counts, labels = percentage_format,  startangle=90, #autopct='%1.0f%%',
                    colors = colors, 
                    textprops={'color':"black", 'size': 6.2, 'weight':'bold'},
                    explode = explode,
                    # increase distance of labels
                    labeldistance = 1.12,
                    )
                    
            
            spines = ['top', 'right', 'left', 'bottom']
            for s in spines:
                ax.spines[s].set_visible(False)

            ax.set_yticks([])
            ax.set_xticks([])

            return ax
        
        if fires_file is not None:
            # if file is string, read it
            if isinstance(fires_file, str): 
                fires = gpd.read_file(fires_file)
                # filter only from: 2008 
                fires[fires_col] = pd.to_datetime(fires[fires_col])
            else:
                fires = fires_file
            
            fires = fires.to_crs(crs)

            annualfire = fires[(fires[fires_col].dt.year == year)] if isinstance(year, int) else fires.copy()
            if season == True:
                months = list(range(4,11)) if month == 1 else list(range(1, 4)) + list(range(11, 13)) 
                annualfire = annualfire[(annualfire[fires_col].dt.month.isin(months))] 
            else:
                annualfire = annualfire[(annualfire[fires_col].dt.month == month)] if month is not None else annualfire
        
        annualsusc = rio.open(hazard_path)

        fig, ax = plt.subplots(figsize=[12,10], dpi = 250)

        allow_fires = False if fires_file is None else allow_fires
        if allow_fires:
            if len(annualfire) != 0:
                annualfire.plot(ax = ax, edgecolor = 'black', linewidth = 0.9, facecolor = 'none')

        month_label = "" if month is None else 'Summer' if month == 1 and season == True else 'Winter' if month == 2 and season == True else month 
        month_labels_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        if isinstance(month_label, int):
            month_label = month_labels_dict[month_label]
        fig, ax = gtras.plot_raster(annualsusc,
                                    add_to_ax = (fig, ax), 
                                    # define the settings for discrete plotting
                                    array_classes = np.arange(-1, 13) + 0.1,
                                    array_colors = ["#99ff9900","#99ff99", "#00ff00", "#006600", "#ffff99",
                                                    "#ffff00", "#cc9900", "#cc99ff", "#9933cc", "#660099",
                                                    "#f55b5b", "#ff0000", "#990000"],
                                    array_names = [ " Not burnable",
                                                "Low intensity surface fires\n with low likelihood",
                                                "Low intensity surface fires\n with medium likelihood",
                                                "Low intensity surface fires\n with high likelihood",
                                                "Medium intensity forest fires\n with low likelihood (broadleaves forests)",
                                                "Medium intensity forest fires\n with medium likelihood (broadleaves forests)",
                                                "Medium intensity forest fires\n with high likelihood (broadleaves forests)",
                                                "High intensity bushfire\n with low likelihood",
                                                "High intensity bushfire\n with medium likelihood",
                                                "High intensity bushfire\n with high likelihood",
                                                "High intensity forest fires\n with low likelihood (coniferous forests)",
                                                "High intensity forest fires\n with medium likelihood (coniferous forests)",
                                                "High intensity forest fires\n with high likelihood (coniferous forests)"
                                            ],
                                    title = f'Hazard {year} {month_label}',
                                    shrink_legend=0.5,
                                )
        allow_hist = False if fires_file is None else allow_hist
        if allow_hist:
            if len(annualfire):

                ax1 = fig.add_axes([xboxmin_hist, yboxmin_hist, 0.18, 0.11])  # left, bottom, width, height
                haz_arr = annualsusc.read(1) 
                                        
            
                stats = gtras.raster_stats_in_polydiss(haz_arr, annualfire, reference_file = hazard_path)
                stats['num_of_burned_pixels'] = stats.num_of_burned_pixels * pixel_to_ha_factor
                print(stats.columns)
                extents = list()
                for _class in stats['class']:
                    extents.append( np.where(haz_arr == _class, 1, 0).sum() * pixel_to_ha_factor)
                stats['extents'] = extents
                stats['percentage'] = stats['num_of_burned_pixels'] / stats['extents'] * 100
                # insert the plot in the same figure and separate axes
                ax1 = histogram(stats, ax1)
        
        if allow_pie:
            try:
                haz_arr.shape
            except:
                haz_arr = annualsusc.read(1) 

            #plot pie chart with classes extent
            ax2 = fig.add_axes([xboxmin_pie, yboxmin_pie, 0.18, 0.18])
            ax2 = pie(ax2, haz_arr, haz_nodata, pixel_to_ha_factor)

        os.makedirs(out_folder, exist_ok = True)
        n = f'haz_plot_{year}{month}.png' if month is not None else f'haz_plot_{year}.png'
        fig.savefig(f'/{out_folder}/{n}', dpi = 200, bbox_inches = 'tight')

        return fig
