import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import deque
from folium.plugins import TimestampedGeoJson
from scipy import stats

from IPython.display import display
from ride_hailing_match import Match
from ride_hailing_location_model import Build_Model
from ride_hailing_simulator import RideHailingENV
from ride_hailing_simulator import Cell
from pyproj import Transformer

def wgs84_to_xy(x_arr: np.ndarray, y_arr: np.ndarray):
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32614')
    x0 = 604082.94
    y0 = 3328141.76
    x_arr_new, y_arr_new = transformer.transform(x_arr, y_arr)
    x_arr_new -= x0
    y_arr_new -= y0
    return x_arr_new.tolist(), y_arr_new.tolist()

def xy_to_wgs84(xy_list):
    transformer = Transformer.from_crs('EPSG:32614', 'EPSG:4326')
    x_new = np.array(xy_list[0]) + 604082.94
    y_new = np.array(xy_list[1]) + 3328141.76
    lat_lon = transformer.transform(x_new, y_new)
    return lat_lon

def xy_to_wgs84_list(xy_list):
    transformer = Transformer.from_crs('EPSG:32614', 'EPSG:4326')
    x_arr = np.array(xy_list[0]) + 604082.94
    y_arr = np.array(xy_list[1]) + 3328141.76
    lat_list, lon_list = transformer.transform(x_arr, y_arr)
    return lat_list.tolist(), lon_list.tolist()

class KSTest:
    def __init__(self) -> None:
        self.env = RideHailingENV(grid_div=3)
        self.origin_data = pd.read_csv('./dataset/rider_data_hr.csv')
        self.hr_time = 15
        self.random_seed = 0
        self.cell = Cell(num_divisions=3)
        self.lat_range, self.lon_range, self.x_range, self.y_range, self.num_divisions = self.cell.pass_info()
        pass

    def ks_location(self, size):
        # Generate simulated rider data
        simulated = self.env.model.gen_riders(size, self.hr_time, self.random_seed)
        simulated_lat, simulated_lon = xy_to_wgs84_list([simulated['x'], simulated['y']])
        simulated = pd.DataFrame({
            'simulated_lat': simulated_lat,
            'simulated_lon': simulated_lon
        })
        
        # Filter simulated data to be within latitude and longitude ranges
        simulated = simulated[
            (simulated['simulated_lat'] >= self.lat_range[0]) & (simulated['simulated_lat'] <= self.lat_range[1]) &
            (simulated['simulated_lon'] >= self.lon_range[0]) & (simulated['simulated_lon'] <= self.lon_range[1])
        ]
        
        # Sample the source data
        origin = self.origin_data[self.origin_data['time_step_index'] == self.hr_time]
        source = origin.sample(n=size)
        source = source[['start_location_lat', 'start_location_long']]
        
        # Filter source data to be within latitude and longitude ranges
        source = source[
            (source['start_location_lat'] >= self.lat_range[0]) & (source['start_location_lat'] <= self.lat_range[1]) &
            (source['start_location_long'] >= self.lon_range[0]) & (source['start_location_long'] <= self.lon_range[1])
        ]
        
        # Perform KS Test
        ks_stat_lat, p_value_lat = stats.ks_2samp(
            simulated['simulated_lat'], 
            source['start_location_lat'],
            method = 'auto'
        )
        ks_stat_lon, p_value_lon = stats.ks_2samp(
            simulated['simulated_lon'], 
            source['start_location_long'],
            method = 'auto'
        )
        
        # Plot CDFs
        plt.figure(figsize=(14, 6))

        # CDF Plot: Latitude
        plt.subplot(1, 2, 1)
        sorted_sim_lat = np.sort(simulated['simulated_lat'])
        sorted_src_lat = np.sort(source['start_location_lat'])
        cdf_sim_lat = np.arange(1, len(sorted_sim_lat)+1) / len(sorted_sim_lat)
        cdf_src_lat = np.arange(1, len(sorted_src_lat)+1) / len(sorted_src_lat)
        
        plt.plot(sorted_src_lat, cdf_src_lat, label='Source Latitude', color='red')
        plt.plot(sorted_sim_lat, cdf_sim_lat, label='Simulated Latitude', color='green')
        plt.xlabel('Latitude')
        plt.ylabel('CDF')
        plt.title('Latitude CDF Kolmogorov-Smirnov Test')
        textstr_lat = f'KS Stat (D): {ks_stat_lat:.3f}, p-value: {p_value_lat*100:.3f}'
        plt.text(
            0.75, 0.065, textstr_lat, transform=plt.gca().transAxes,
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
        )
        plt.legend([f'Source Latitude',
                    f'Simulated Latitude'], 
                    loc='best')

        # CDF Plot: Longitude
        plt.subplot(1, 2, 2)
        sorted_sim_lon = np.sort(simulated['simulated_lon'])
        sorted_src_lon = np.sort(source['start_location_long'])
        cdf_sim_lon = np.arange(1, len(sorted_sim_lon)+1) / len(sorted_sim_lon)
        cdf_src_lon = np.arange(1, len(sorted_src_lon)+1) / len(sorted_src_lon)
        
        plt.plot(sorted_src_lon, cdf_src_lon, label='Source Longitude', color='red')
        plt.plot(sorted_sim_lon, cdf_sim_lon, label='Simulated Longitude', color='green')
        plt.xlabel('Longitude')
        plt.ylabel('CDF')
        plt.title('Longitude CDF Kolmogorov-Smirnov Test')
        textstr_lon = f'KS Stat (D): {ks_stat_lon:.3f}, p-value: {p_value_lon*100:.3f}'
        plt.text(
            0.75, 0.065, textstr_lon, transform=plt.gca().transAxes,
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
        )
        plt.legend([f'Source Longitude',
                    f'Simulated Longitude'], 
                    loc='best')
        
        plt.tight_layout()
        plt.show()
        
        return simulated, source
    
    def ks_demand(self, size):
        origin = self.origin_data
        # Filter source data to be within latitude and longitude ranges
        origin = origin[
            (origin['start_location_lat'] >= self.lat_range[0]) & (origin['start_location_lat'] <= self.lat_range[1]) &
            (origin['start_location_long'] >= self.lon_range[0]) & (origin['start_location_long'] <= self.lon_range[1])
        ]

        # Generate simulated rider data
        simulated = origin.sample(n=size)
        simulated = simulated['time_step_index']
        simulated_1 = origin.sample(n=size)
        simulated_1 = simulated_1['time_step_index']
        
        # Sample the source data
        source = origin.sample(n=size)
        source = source['time_step_index']
        source_1 = origin.sample(n=size)
        source_1 = source_1['time_step_index']
    
        # Perform KS Test
        ks_stat, p_value = stats.ks_2samp(simulated, source, method = 'auto')
        ks_stat_1, p_value_1 = stats.ks_2samp(simulated_1, source_1, method = 'auto')
        
        # Plot CDFs
        plt.figure(figsize=(14, 6))

        # CDF Plot: Demand, test 1
        plt.subplot(1, 2, 1)
        sorted_sim_demand = np.sort(simulated)
        sorted_src_demand = np.sort(source)
        cdf_sim_demand = np.arange(1, len(sorted_sim_demand)+1) / len(sorted_sim_demand)
        cdf_src_demand = np.arange(1, len(sorted_src_demand)+1) / len(sorted_src_demand)
        
        plt.plot(sorted_src_demand, cdf_src_demand, label='Source Hourly Demands', color='red')
        plt.plot(sorted_sim_demand, cdf_sim_demand, label='Simulated Hourly Demands', color='green')
        plt.xlabel('Hour')
        plt.ylabel('Demand CDF')
        plt.title('Hourly Demand CDF Kolmogorov-Smirnov Test-1')
        textstr_lat = f'KS Stat (D): {ks_stat:.3f}, p-value: {p_value:.3f}'
        plt.text(
            0.75, 0.065, textstr_lat, transform=plt.gca().transAxes,
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
        )
        plt.legend([f'Source Hourly Demands',
                    f'Simulated Hourly Demands'], 
                    loc='best')
        
        # CDF Plot: Demand, test 2
        plt.subplot(1, 2, 2)
        sorted_sim_demand = np.sort(simulated_1)
        sorted_src_demand = np.sort(source_1)
        cdf_sim_demand = np.arange(1, len(sorted_sim_demand)+1) / len(sorted_sim_demand)
        cdf_src_demand = np.arange(1, len(sorted_src_demand)+1) / len(sorted_src_demand)
        
        plt.plot(sorted_src_demand, cdf_src_demand, label='Source Hourly Demands', color='red')
        plt.plot(sorted_sim_demand, cdf_sim_demand, label='Simulated Hourly Demands', color='green')
        plt.xlabel('Hour')
        plt.ylabel('Demand CDF')
        plt.title('Hourly Demand CDF Kolmogorov-Smirnov Test-2')
        textstr_lat = f'KS Stat (D): {ks_stat_1:.3f}, p-value: {p_value_1:.3f}'
        plt.text(
            0.75, 0.065, textstr_lat, transform=plt.gca().transAxes,
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
        )
        plt.legend([f'Source Hourly Demands',
                    f'Simulated Hourly Demands'], 
                    loc='best')
        
        return simulated, source