"""
# Use this file:
import ride_hailing_location_model as rh_model
model = rh_model.Build_Model()
rider_model, driver_model = model.get_model()

# How to test the fitted model
sample = model.sample_from_model(rider_model['23'], 3000)
fig = plt.figure(figsize=(10,7))
plt.scatter(sample[f'sampled_location_lat'], sample[f'sampled_location_long'], alpha=0.7)
plt.grid()
plt.show()
"""


import datetime
import folium

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from folium.plugins import HeatMap
from sklearn.neighbors import KernelDensity


class Dataset_Info:
    def show_heatmap(coord_data, keyword='start_location', zoom=9, data_format='dataframe'):
        m = folium.Map(location=(30.2672, -97.7431), zoom_start=zoom)
        if data_format == 'dataframe':
            data = list(coord_data.groupby([f'{keyword}_lat', f'{keyword}_long']).groups.keys())
        if data_format == 'ndarray':
            data = coord_data
        heatmap = HeatMap(data, radius=14)
        heatmap.add_to(m)

        return m

    def draw_hist(data, title):
        plt.figure(figsize=(15,2))
        plt.hist(data, density=True, bins=500) 
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title(title)
        plt.grid()
        plt.show()

        pass

    def draw_scatter(data, keyword, title, sample_plot=True, sample_scale=1000, get_sample=False):
        fig = plt.figure(figsize=(10,7))
        if sample_plot == True:
            sample_idx = np.random.choice(data.shape[0], sample_scale, replace=False)
            sampled_reverse = data.loc[sample_idx]
            sampled_data = sampled_reverse[[f'{keyword}_lat', f'{keyword}_long']]
            plt.scatter(sampled_data[f'{keyword}_lat'], sampled_data[f'{keyword}_long'], alpha=0.7)
        else:
            plt.scatter(data[f'{keyword}_lat'], data[f'{keyword}_long'], alpha=0.7)
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title(title)
        plt.grid() 
        plt.show()

        if get_sample == True:
            return sampled_data


class Build_Model:
    def __init__(self) -> None:
        self.rider_dataset_hr = pd.read_csv('./dataset/rider_data_hr.csv')
        self.driver_dataset_hr = pd.read_csv('./dataset/driver_data_hr.csv')
        self.lat_range = [30.10, 30.41] # location gennerating range
        self.lon_range = [-97.88, -97.57] # location gennerating range

        pass
    
    def data_slice(self, dataset):
        start_loc = dataset[["RIDE_ID", "started_on", "start_location_lat", "start_location_long"]]
        end_loc = dataset[["RIDE_ID", "completed_on", "end_location_lat", "end_location_long"]]

        return start_loc, end_loc
    
    def data_filter(self):
        data = pd.read_csv('./dataset/all_data.csv')
        lat_range = self.lat_range # location gennerating range
        lon_range = self.lon_range # location gennerating range
        data = data.drop(data[(data['start_location_lat']<lat_range[0]) | (data['start_location_lat']>lat_range[1])].index)
        data = data.drop(data[(data['start_location_long']<lon_range[0]) | (data['start_location_long']>lon_range[1])].index)
        data = data.drop(data[(data['end_location_lat']<lat_range[0]) | (data['end_location_lat']>lat_range[1])].index)
        data = data.drop(data[(data['end_location_long']<lon_range[0]) | (data['end_location_long']>lon_range[1])].index)
        start_loc, end_loc = self.data_slice(data)
        rider_dataset_hr = self.add_step_info(start_loc, keyword='started_on', step_size=60)
        driver_dataset_hr = self.add_step_info(end_loc, keyword='completed_on', step_size=60)
        rider_dataset_hr.to_csv('./dataset/rider_data_hr.csv')
        driver_dataset_hr.to_csv('./dataset/driver_data_hr.csv')

        pass

    def add_step_info(self, data, keyword, step_size=60):
        # keyword: which column contains time data, step_size: how many minuetes per time step
        step_dataset = data.copy()
        step_dataset[f'{keyword}'] = pd.to_datetime(step_dataset[f'{keyword}'])
        step_dataset[f'{keyword}'] = step_dataset[f'{keyword}'].dt.time
        step_dataset.insert(step_dataset.shape[1], 'time_step_index', 1)
        number_step = int(1440 / step_size)

        start_time = datetime.datetime.strptime('00:00:00', '%H:%M:%S')
        time_debug = datetime.datetime.strptime('1900-01-02 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.timedelta(minutes=step_size)

        for i in range(number_step):
            if start_time == time_debug:
                end_time = start_time + datetime.timedelta(minutes=step_size) - datetime.timedelta(seconds=1)
            else:
                end_time = start_time + datetime.timedelta(minutes=step_size)
            start_time_f = start_time.time()
            end_time_f = end_time.time()
            step_index = i + 1
            step_dataset.loc[(step_dataset[f'{keyword}'] > start_time_f) & (step_dataset[f'{keyword}'] <= end_time_f), 'time_step_index'] = step_index
            start_time = datetime.datetime.strptime(str(end_time_f), '%H:%M:%S')

        return step_dataset

    def sample_from_model(self, model: object, num_sample: int, seed = None) -> list:
        if seed == None:
            sample = model.sample(num_sample)
        else:
            sample = model.sample(num_sample, random_state=seed)
        return sample

    def fit_model(self, data: pd.DataFrame, keyword: str, band_width:float = 0.0035) -> dict:
        # keyword: 'driver' or 'rider'
        model_set = {}

        if keyword == 'driver':
            index_key = 'end_location'
        else:
            index_key = 'start_location'

        for time in range(1,25):
            data_hr = data[data['time_step_index']==time]
            xy_train  = np.vstack([data_hr[f'{index_key}_lat'], data_hr[f'{index_key}_long']]).T
            kde_skl = KernelDensity(kernel='gaussian', bandwidth=band_width)
            model_set[f'{time}'] = kde_skl.fit(xy_train)

        return model_set
    
    def get_model(self):
        rider_model = self.fit_model(self.rider_dataset_hr, keyword='rider')
        driver_model = self.fit_model(self.driver_dataset_hr, keyword='driver')
        
        return rider_model, driver_model