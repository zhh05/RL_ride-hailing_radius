"""
# Ride_hailing env
This is the file for defining the simulator for the ride-hailing environment.
Functions defined in this file can be used for reinforcement learning.
This is part of the master thesis project:
Optimising matching radius for a ride-hailing system.

# Use this .py script:
env = RideHailingENV(grid_div=2)
ob_rider, ob_driver, done = env.reset()
ob_rider, ob_driver, reward = env.step(action, time_step, rend_step=False)

# Test this environment for one step:
env = RideHailingENV(grid_div=2)
ob_rider, ob_driver, done = env.reset()
action = np.full(4, 800)
ob_rider, ob_driver, reward, done = env.step(action, hr_time=1, rend_step=True)
print(reward, done)

# Test this environment for one episode (5 hours):
env = RideHailingENV(grid_div=2)
ob_rider, ob_driver, done = env.reset()
time_step, ep_reward, step_count = 1, 0, 0
action = np.full(4, 800)
while not done:
    ob_rider, ob_driver, reward, done = env.step(action, time_step, rend_step=False)
    ep_reward += reward
    step_count += 1
print(f'{ep_reward} | {step_count}')
"""


import gym
import folium

import numpy as np
import pandas as pd

from h3 import h3
from gym import spaces
from folium.features import DivIcon
from IPython.display import display
from ride_hailing_match import Match
from ride_hailing_location_model import Build_Model
from pyproj import Transformer


Global_Resolution = 7 # change the resolution here, determines the number of cells
# !!!! change


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


class Cell:
    """Gennerate cells
    """
    def __init__(self, num_divisions) -> None:
        self.lat_range = np.array([30.18, 30.32]) # Austin latitude range
        self.lon_range = np.array([-97.81, -97.65]) # Austin longitude range
        self.x_range, self.y_range = wgs84_to_xy(self.lat_range, self.lon_range)
        self.num_divisions = num_divisions # how many part lat and lon are divided
        pass
   
    def pass_info(self):
        return self.lat_range, self.lon_range, self.x_range, self.y_range, self.num_divisions
    
    def get_cells(self, display_map: bool = False) -> list:
        number =  self.num_divisions ** 2
        cells = np.arange(number)

        if display_map == False:
            return cells
        
        else:
            lat_step = (self.lat_range[1] - self.lat_range[0]) / self.num_divisions
            lon_step = (self.lon_range[1] - self.lon_range[0]) / self.num_divisions
            m = folium.Map(location=[(self.lat_range[0] + self.lat_range[1]) / 2, (self.lon_range[0] + self.lon_range[1]) / 2], zoom_start=13)
            for i in range(self.num_divisions):
                for j in range(self.num_divisions):
                    lat_start = self.lat_range[0] + i * lat_step
                    lat_end = lat_start + lat_step
                    lon_start = self.lon_range[0] + j * lon_step
                    lon_end = lon_start + lon_step
                    grid_number = i * self.num_divisions + j
                    # Draw the grid
                    folium.Rectangle(
                        bounds=[[lat_start, lon_start], [lat_end, lon_end]],
                        color='blue',
                        fill=True,
                        fill_opacity=0.1
                    ).add_to(m)
                    # Add grid number
                    folium.Marker(
                        location=[(lat_start + lat_end) / 2, (lon_start + lon_end) / 2],
                        icon=folium.DivIcon(html=f'<div style="font-size: 12pt">{grid_number}</div>')
                    ).add_to(m)
            display(m)
            return cells

    def get_cell_id_wgs84(self, lat, lon):
        if not (self.lat_range[0] <= lat <= self.lat_range[1]) or not (self.lon_range[0] <= lon <= self.lon_range[1]): # check if in the range
            return None
        lat_step = (self.lat_range[1] - self.lat_range[0]) / self.num_divisions
        lon_step = (self.lon_range[1] - self.lon_range[0]) / self.num_divisions
        lat_index = min(int((lat - self.lat_range[0]) / lat_step), self.num_divisions - 1)
        lon_index = min(int((lon - self.lon_range[0]) / lon_step), self.num_divisions - 1)
        grid_number = lat_index * self.num_divisions + lon_index
        return grid_number
    
    def get_cell_id_xy(self, x_list, y_list):
        cell_ids = []
        x_step = (self.x_range[1] - self.x_range[0]) / self.num_divisions
        y_step = (self.y_range[1] - self.y_range[0]) / self.num_divisions

        for x, y in zip(x_list, y_list):
            x_index = min(int((x - self.x_range[0]) / x_step), self.num_divisions - 1)
            y_index = min(int((y - self.y_range[0]) / y_step), self.num_divisions - 1)
            grid_number = x_index + y_index * self.num_divisions
            cell_ids.append(grid_number)
        
        return cell_ids
        

class Gen_Model:
    """Sample locations from fitted model for riders and drivers in the map

    This class is to generate locations for riders and drivers based on the given
    distribution of their locations. This default model is estimated with Kernel 
    Density Estimation (KDE). The generated location is given in the format of 
    a pandas DataFrame, each row represents a unique rider/driver. Information
    given in a row includes rider/driver's ID, H3 code, longitude and latitude.

    Attributes:
        model: an instance of Build_Model class containing models for riders and drivers
        rider_model: a dictionary for 24 KDE distributions, describing the locational and timely distribution of riders
        driver_model: a dictionary for 24 KDE distributions, describing the locational and timely distribution of drivers
    """
    def __init__(self, num_div) -> None:
        self.cell = Cell(num_div)
        self.cell_ids = self.cell.get_cells(False)
        self.model = Build_Model()
        self.rider_model, self.driver_model = self.model.get_model()

    def gen_drivers(self, number_of_drivers: int, hr_time: int, seed: int = None):
        """Sample locations for drivers
        
        Sample multiple locations for drivers based on the given locational distribution 
        of drivers. The default distribution model is KDE.

        Parameters:
            number_of_drivers: an int, indicating how many drivers are generated.
            hr_time: an int, the value is the hour of the day, indicating which distribution model will be used.
            resolution: an int, determines the number of cells in the map.

        Returns:
            driver_df: a pandas DataFrame, including information of generated drivers. 
        """
        if seed != None:
            driver_locations = self.model.sample_from_model(self.driver_model[f'{hr_time}'], number_of_drivers, seed) # dtype = numpy ndarray
        else:
            driver_locations = self.model.sample_from_model(self.driver_model[f'{hr_time}'], number_of_drivers) # dtype = numpy ndarray
        driver_ids = []
        cell_ids = []

        for driver_id, geo_info in enumerate(driver_locations):
            cell_id = self.cell.get_cell_id_wgs84(geo_info[0], geo_info[1])
            while cell_id is None:
                geo_info = self.model.sample_from_model(self.rider_model[f'{hr_time}'], 1)[0] # dtype = numpy ndarray
                cell_id = self.cell.get_cell_id_wgs84(geo_info[0], geo_info[1])
                driver_locations[driver_id] = geo_info
                #print('Re-sampled a driver!')
            cell_ids.append(cell_id)
            driver_ids.append(driver_id)

        x_list, y_list = wgs84_to_xy(driver_locations.T[0], driver_locations.T[1])

        driver_df = pd.DataFrame({'driver_id':driver_ids, 'cell_id':cell_ids, 'x':x_list, 'y':y_list, 'statue': 1, 'idle_time': 0})
        # driver is avliable: 'statue' = 1, unavliable: 'statue' = 0
        # driver_df[['driver_id', 'cell_id']] = driver_df[['driver_id', 'cell_id']].astype(int) # set data type to int

        return driver_df

    def gen_riders(self, number_of_riders: int, hr_time: int, seed: int = None):
        """Sample locations for riders
        
        Sample multiple locations for riders based on the given locational distribution 
        of riders. The default distribution model is KDE.

        Parameters:
            number_of_riders: an int, indicating how many riders are generated.
            hr_time: an int, the value is the hour of the day, indicating which distribution model will be used.
            resolution: an int, determines the number of cells in the map.

        Returns:
            rider_df: a pandas DataFrame, including information of generated riders. 
        """
        if seed != None:
            rider_locations = self.model.sample_from_model(self.rider_model[f'{hr_time}'], number_of_riders, seed) # dtype = numpy ndarray
        else:
            rider_locations = self.model.sample_from_model(self.rider_model[f'{hr_time}'], number_of_riders) # dtype = numpy ndarray
        rider_ids = []
        cell_ids = []
        
        for rider_id, geo_info in enumerate(rider_locations):
            cell_id = self.cell.get_cell_id_wgs84(geo_info[0], geo_info[1])
            while cell_id is None:
                geo_info = self.model.sample_from_model(self.rider_model[f'{hr_time}'], 1)[0] # dtype = numpy ndarray
                cell_id = self.cell.get_cell_id_wgs84(geo_info[0], geo_info[1])
                rider_locations[rider_id] = geo_info
                #print('Re-sampled a rider!')
            cell_ids.append(cell_id)
            rider_ids.append(rider_id)

        x_list, y_list = wgs84_to_xy(rider_locations.T[0], rider_locations.T[1])

        rider_df = pd.DataFrame({'rider_id':rider_ids, 'cell_id':cell_ids, 'x':x_list, 'y':y_list, 'time_step_in_pool': 1})
        # rider_df[['rider_id', 'cell_id', 'time_step_in_pool']] = rider_df[['rider_id', 'cell_id', 'time_step_in_pool']].astype(int) # set data type to int

        return rider_df


class RideHailingENV(gym.Env):
    """Simulation environment for project optimising matching radius for ride-hailing system

    This class is the main simulator for the master thesis project optimising matching radius 
    for a ride-hailing system with reinforcement learning. The project is carried out in TU Delft. 
    This simulator is built base on the geographical information of Austin, Texas, USA. It intake
    continous matching radius as the action, and the reward is the total net profit made by the 
    system within a day.

    GOOD action for example:
    action = np.ones(36)*800
    action[[5, 13, 14, 15, 19, 20, 21, 26, 27]] = 400

    Attributes:
     lower_bound - lower bound of action space, minimum matching radius, unit is meters.
     upper_bound - upper bound of action space, maximum matching radius, unit is meters.
     model - make an instance of Gen_Model class, to generate riders and drivers for the simulator.
     match - make an instance of Match class, to run the matching algorithm.
     radius_initial - initial matching radius when reset the environment, unit is meters.
     driver_num_ini - initial number of drivers, can be changed if set dynamic.
     rider_num_ini - initial number of riders, rider number is changing among different steps.
     fuel_unit_price - average travelling fuel cost per vehicle per kilometer in the US, the unit is US dollars.
     time_window - time interval between every two matching process (Uber Batched matching), fixed among all steps, unit is minutes.
     total_reward - total reward for the intake action.
     gen_rate_rider - overall generating rate of riders, number of riders per time-window.
     gen_rate_driver - active if vehicle number are set dynamic, number of new drivers per time-window.
     ride_price - average ride price in Austin urban area, the value is estimated from Uber ride data in 2022.
     rider_patience - the maximum number of steps a rider can stay in the matching pool.
     p_unmatch_rider - penalty per unmatched rider, the value is cauculated base on the probability of losing a potential ride.
     action_space - defines the numerical range of intake actions.
     observation_space - defines the numerical range of overall observations.
     sub_observation_space - defines the numerical range of observations within a cell.

    Distance cost explain:
     distance cost = car-buying cost + car repair and maintanance cost + fuel cost
     car repair and maintanance cost = change tire per 200000km &1000 + change motor oil per 200000km $1200
     car-buying cost: average car price in the us is $22000
     fuel cost: averge fuel cost $1.3/L * averge fuel comsuption 9.3L/100km = $12.1/100km = $0.12/km
     total = $11.65/km
    """
    def __init__(self, grid_div, time_window: int = 0.25) -> None:

        lower_bound = 50
        upper_bound = 6000
        self.num_divi = grid_div
        self.cell = Cell(self.num_divi)
        self.lat_range, self.lon_range, self.x_range, self.y_range, self.num_divisions = self.cell.pass_info()
        self.model = Gen_Model(self.num_divi)
        self.match = Match()
        self.seed_ini = None

        self.cell_ids = self.cell.get_cells(False)
        self.cell_num = np.size(self.cell_ids)
        self.radius_initial = 500
        self.driver_num_ini = 100
        self.rider_num_ini = 15
        self.distance_cost = 13.26 * 0.001 # this is a combination of fuel cost, car-buying and car-repair cost, unit is per meter
        self.fuel_cost = 1.93 * 0.001
        self.time_window = time_window
        self.total_reward = 0
        self.gen_rate_rider = 10
        self.gen_rate_driver = 10
        self.ride_price = 23.92
        self.rider_patience = 5 # minutes
        self.p_rider_left = 0.1 # punishment
        self.p_unmatch_rider = 5
        self.simulation_time = 30

        self.drivers = None
        self.riders = None
        self.drivers_tmp = None
        self.riders_tmp = None

        self.score_distance = lambda distance, max_distance=1200: max(0, min(1, 1 - distance / max_distance))
        self.score_radius = lambda distance, max_distance=6000: max(0, min(1, distance / max_distance))
        self.multi_task_weight_factor = [0.4, 0.6, 0] # order fulfillment rate, average pickup distance, driver utilization rate
        #self.multi_task_weight = [0.6, 0.4] # order fulfillment rate, average pickup distance

        self. rider_patience_step = self.rider_patience / self.time_window

        self.action_space = spaces.Box(
                low=np.array(lower_bound*np.ones(self.cell_num)), 
                high=np.array(upper_bound*np.ones(self.cell_num)),
                dtype=np.int32
                )
        self.observation_space = spaces.Discrete(self.num_divi ** 2) # updated

        self.step_count = 0
        self.max_step = self.simulation_time / self.time_window

    def reset(self, time_ini: int = 1) -> np.array:
        """
        reset the environment for the first step in every episode.

        Parameters:
         time_ini -  set the initial time to 0-1 hour of a day.

        Returns:
         returns the initial matching radius.
        """
        self.random_seed = self.seed_ini
        self.drivers = self.model.gen_drivers(self.driver_num_ini, time_ini, self.random_seed)
        self.drivers_in_service = np.zeros(self.driver_num_ini, dtype=np.int32)
        self.drivers['driver_id'] = np.arange(self.drivers.shape[0])

        self.riders = self.model.gen_riders(self.rider_num_ini, time_ini, self.random_seed)
        self.riders['rider_id'] = np.arange(self.riders.shape[0])

        self.total_request_num = len(self.riders)
        self.matched_ride_num = 0
        self.observe = np.zeros([self.num_divi ** 2, 2]) # reset observation space

        done = False
        self.step_count = 0

        rider_vec = self.riders[['x', 'y']].values
        driver_vec = self.drivers[['x', 'y']].values
        self.dis_matrix = self.__vector_dis(rider_vec, driver_vec)

        rider, driver = self.get_observe()

        return rider, driver, done #, self.riders, self.drivers
    
    def step(self, radius: np.array, hr_time: int, rend_step: bool = False) -> tuple[float, dict, list]:
        """
        The main process of a step.

        Parameters:
         radius - matching radius for each cell.
         hr_time - the hourly time step in a day, determines location distribution of riders and drivers.
         rend_step - visualize one step if set to True.

        Returns:
         the reward of one step and matched pairs within this step. 
        """

        #print(self.x_range, self.y_range, self.drivers)
        #self.random_seed += 1 # update seed
        self.riders_tmp = self.riders.copy()
        self.drivers_tmp = self.drivers.copy()

        #print(len(self.riders), len(self.drivers[self.drivers['statue']==1]))

        rider_vec = self.riders[['x', 'y']].values
        driver_vec = self.drivers[['x', 'y']].values

        # get the distance matrix and matching pool
        self.dis_matrix = self.__vector_dis(rider_vec, driver_vec)
        cell_ids = self.riders['cell_id']
        r_radius = radius[cell_ids]

        # get the matching pool
        pool = self.__get_pool(self.dis_matrix, r_radius)

        # matching process
        match_statue = self.__match(pool)
        reward = self.__execute_match(match_statue, pool, radius)
        
        if rend_step:
            state = [self.riders_tmp, self.drivers_tmp]
            self.render(state, radius, match_statue)

        self.riders, self.drivers, reward, done = self.__state_transit(hr_time, reward, match_statue)

        # give terminal reward
        if done == True:
            reward += 0 #100
        else:
            reward -= 0 #len(self.riders)
            self.step_count += 1

        rider, driver = self.get_observe()
        return rider, driver, reward, done
    
    def get_observe(self): # observation is number of riders/drivers in each cell
        #driver_info = self.drivers[['x', 'y', 'statue']].values.flatten().tolist()
        #observe = np.hstack([rider_counts, driver_info])

        # new version observation, according to ref: cell num, time_step, previoud radius, number of drivers, number of riders
        rider_counts = self.riders['cell_id'].value_counts().sort_index().reindex(range(self.cell_num), fill_value=0).to_numpy()
        supply_driver = self.drivers[self.drivers['statue']==1]
        driver_counts = supply_driver['cell_id'].value_counts().sort_index().reindex(range(self.cell_num), fill_value=0).to_numpy()
        #self.observe[:,0] = rider_counts
        #self.observe[:,1] = driver_counts

        avg_distance = self.dis_matrix

        return rider_counts, driver_counts#, avg_distance
    
    def check_done(self):
        if self.step_count >= self.max_step -1: #or self.riders.shape[0] == 0:
            done = True
        else:
            done = False
        return done

    def __get_pool(self, dis_matrix: list, radius_set: int) -> np.ndarray:
        """
        form a matching pool for all the riders and available drivers whithin the matching radius.

        Parameters:
            riders - locations, numbers of all the riders.
            drivers - locations, numbers of all the drivers.
            radius - matching radius for each cell, riders in the same cell have the same matching radius.

        Returns:
            returns a list consist all the possible matches and the distance between them.
        """
        match_pool = []
        for i in range(dis_matrix.shape[0]):
            sub_pool = []
            radius_rider = radius_set[i]
            driver_1 = np.where(dis_matrix[i] <= radius_rider)[1]
            driver_2 = self.drivers.index[self.drivers['statue'] == 1]
            driver = list(np.intersect1d(driver_1, driver_2))
            rider = list(np.ones(np.size(driver), dtype=int)*i)
            dis = dis_matrix[i, driver].tolist()[0]
            sub_pool.extend([rider])
            sub_pool.extend([driver])
            sub_pool.extend([dis])
            sub_pool = list(map(list, zip(*sub_pool)))
            match_pool.extend(sub_pool)
        return match_pool
            
    def __match(self, pool: list): # MM: Maximum Matching, OM: Optimised Matching
        """
        excute matching algorithm to find the optimal match for the given matching pool.

        Parameters:
         pool - matching pool with distance. 

        Returns:
         match statue with matched pairs and their distance. 
        """
        matched_pairs = self.match.match(pool, method='Munkres')
        return matched_pairs
    
    def __execute_match(self, match_statue:list, match_pool: list, radius: np.array) -> tuple[float, tuple]:
        """
        apply the matched pairs to the map, update riders and drivers, observe reward and penalty.

        Parameters:
        riders - locations, numbers of all the riders.
        drivers - locations, numbers of all the drivers.
        match_statue - matched pair of riders and drivers with the distance between them.

        Returns:
        reward - the net monetary profit made from the ride-hailing system within a step.
        pool_next - next state of the environment after taking the action.
        """
        # calculate total distance and rewards
        reward = 0 #- 200 * self.distance_cost
        self.matched_ride_num += len(match_statue)
        complete_rate = self.matched_ride_num / self.total_request_num

        if match_statue:
            aver_distance = sum(distance for _, _, distance in match_statue) / len(match_statue)
            riders_to_drop, drivers_to_drop = zip(*[(rider, driver) for rider, driver, _ in match_statue])
            riders_to_drop, drivers_to_drop = list(riders_to_drop), list(drivers_to_drop)
            reward_match_rate = len(match_statue) / len(self.riders)
            reward_driver_ult = len(match_statue) / len(match_pool)
        else:
            aver_distance = 0 # in case for tricky policies
            riders_to_drop, drivers_to_drop = [], []
            reward_match_rate = 0
            reward_driver_ult = 0
                 
        reward_distance = self.score_distance(aver_distance)

        #print(complete_rate, reward_distance, reward_driver_ult)

        #reward = complete_rate * self.multi_task_weight_factor[0] + reward_distance * self.multi_task_weight_factor[1] + reward_driver_ult * self.multi_task_weight_factor[2]
        reward = complete_rate * self.multi_task_weight_factor[0] - self.score_radius(sum(radius) / self.cell_num) * self.multi_task_weight_factor[1]

        self.riders = self.riders.drop(riders_to_drop)
        self.drivers_in_service[drivers_to_drop] += 2
        self.drivers.loc[drivers_to_drop, 'statue'] = 0
        self.drivers.loc[drivers_to_drop, 'idle_time'] = 0

        self.riders['time_step_in_pool'] += 1
        #reward -= self.p_rider_left * self.riders[self.riders['time_step_in_pool']>self.rider_patience].shape[0]
        self.riders = self.riders.drop(self.riders[self.riders['time_step_in_pool']>self.rider_patience_step].index) # inpatient riders quit the matching pool

        # calculate total punishment for unmatched riders and drivers
        # punishment = self.p_unmatch_rider * len(self.riders) #+ self.fuel_cost * len(self.drivers) * 250 # average idling dis per time window

        # calculate net reward
        net_reward = reward #- punishment

        return net_reward
    
    def __vector_dis(self, rider_vec, driver_vec):
        m = np.shape(rider_vec)[0]
        n = np.shape(driver_vec)[0]
        M = np.dot(rider_vec, driver_vec.T)
        H = np.tile(np.matrix(np.square(rider_vec).sum(axis=1)).T,(1,n))
        K = np.tile(np.matrix(np.square(driver_vec).sum(axis=1)),(m,1))
        return np.sqrt(-2 * M + H + K)
    
    def __state_transit(self, hr_time: int, reward: float, match_statue: list) -> dict: 
        """
        update the current state and give the state of the next step.
        v1_update: do not gennerate new riders, terminal is all the riders are matched or left

        Parameters:
         state - the current state, locations of riders and drivers.
         hr_time - hourly time of a day, this is used to generate new riders and drivers.

        Returns:
         returns the locations of riders and drivers for the next step.
        """
        ride_num = np.size(match_statue)

        # update riders
        done = self.check_done()
        if done == True:
            return self.riders, self.drivers, reward, done

        rider_size = self.gen_rate_rider
        self.total_request_num += rider_size
        new_riders = self.model.gen_riders(rider_size, hr_time, self.random_seed)
        rider_next = pd.concat((self.riders, new_riders), axis=0)

        self.drivers.loc[self.drivers_in_service == 0, 'statue'] = 1 # drivers finished ride

        # update drivers - driver reposition
        self.drivers.loc[self.drivers['statue'] == 1, 'idle_time'] += 1 # update idle time
        condition = (self.drivers['idle_time'] == 20) & (self.drivers['statue'] == 1)
        self.drivers.loc[condition, 'x'] += np.random.choice([-800, 800], size=condition.sum())
        self.drivers.loc[condition, 'y'] += np.random.choice([-800, 800], size=condition.sum())
        self.drivers.loc[condition, 'idle_time'] = 0 # reset idle time

        # update drivers - driver idling
        self.drivers_in_service[self.drivers_in_service != 0] -= 1
        self.drivers['x'] += np.random.uniform(-400, 400, size=self.drivers.shape[0])
        self.drivers['y'] += np.random.uniform(-400, 400, size=self.drivers.shape[0])

        # check latitude and longitude border
        self.drivers['x'] = np.clip(self.drivers['x'], self.x_range[0], self.x_range[1]) # check latitude range
        self.drivers['y'] = np.clip(self.drivers['y'], self.y_range[0], self.y_range[1]) # check longitude range

        # update cell ids for drivers
        self.drivers['cell_id'] = self.cell.get_cell_id_xy(self.drivers['x'], self.drivers['y']) 

        #rider_next = rider_next.reset_index(drop=True)
        self.riders = rider_next.reset_index(drop=True)
        self.drivers = self.drivers.reset_index(drop=True)

        # re-index drivers and riders
        driver_index = self.drivers.shape[0]
        self.drivers['driver_id'] = np.arange(driver_index)
        rider_index = self.riders.shape[0]
        self.riders['rider_id'] = np.arange(rider_index)

        return self.riders, self.drivers, reward, done

    def render(self, state: tuple, radius_set: dict, match_statue: list, color_set: tuple = ['red', 'blue'], folium_map=None) -> None:
        """
        visualise the state and action for one step, red circle is matching range (within matching radius),
        green lines are the links for matched pairs.

        Parameters:
         state - the current state, locations of riders and drivers.
         radius_set - matching radius for each cell.
         match_statue - matched pair of riders and drivers with the distance between them.
         folium_map - map object.
        """

        riders = state[0]
        drivers = state[1]
        drivers = drivers[drivers['statue'] == 1]
        drivers.reset_index(drop=True, inplace=True)
        matched_riders = []
        matched_drivers = []
        if match_statue != []:
            matched_riders = pd.DataFrame(match_statue)[0].to_list()
            matched_drivers = pd.DataFrame(match_statue)[1].to_list()

        matched_rider_location = {}
        matched_driver_location = {}

        lat_step = (self.lat_range[1] - self.lat_range[0]) / self.num_divisions
        lon_step = (self.lon_range[1] - self.lon_range[0]) / self.num_divisions
        m = folium.Map(location=[(self.lat_range[0] + self.lat_range[1]) / 2, (self.lon_range[0] + self.lon_range[1]) / 2], zoom_start=13)
        for i in range(self.num_divisions):
            for j in range(self.num_divisions):
                lat_start = self.lat_range[0] + i * lat_step
                lat_end = lat_start + lat_step
                lon_start = self.lon_range[0] + j * lon_step
                lon_end = lon_start + lon_step
                grid_number = i * self.num_divisions + j
                # Draw the grid
                folium.Rectangle(
                    bounds=[[lat_start, lon_start], [lat_end, lon_end]],
                    color='blue',
                    fill=True,
                    fill_opacity=0.1
                ).add_to(m)
                # Add grid number
                folium.Marker(
                    location=[(lat_start + lat_end) / 2, (lon_start + lon_end) / 2],
                    icon=folium.DivIcon(html=f'<div style="font-size: 18pt">{grid_number}</div>')
                ).add_to(m)
       
        # add driver markers
        for i in range(drivers.shape[0]):
            driver_wgs = xy_to_wgs84([drivers.loc[i]['x'], drivers.loc[i]['y']])
            folium.Marker(
                location=driver_wgs,
                icon=folium.Icon(
                    color=color_set[0],
                    prefix='fa',
                    icon='car'
                    )
                ).add_to(m)
            if drivers.loc[i]['driver_id'] in matched_drivers:
                matched_driver_location[f'{int(drivers.loc[i]["driver_id"])}'] = driver_wgs
          
        # add rider markers and matching radius
        for j in range(riders.shape[0]):
            rider_wgs = xy_to_wgs84([riders.loc[j]['x'], riders.loc[j]['y']])
            folium.Marker(
                location=rider_wgs,
                icon=folium.Icon(
                    color=color_set[1],
                    prefix='fa',
                    icon='male'
                    )
                ).add_to(m)
            
            folium.Circle(
                    radius=float(radius_set[int(riders.loc[j]['cell_id'])]),
                    location=rider_wgs,
                    color="red",
                    weight=1,
                    fill=True,
                    fill_opacity=0.1
                ).add_to(m)
            if riders.loc[j]['rider_id'] in matched_riders:
                matched_rider_location[f'{int(riders.loc[j]["rider_id"])}'] = rider_wgs
       
        for rider, driver, dis in match_statue:
            folium.PolyLine(
                locations=[matched_rider_location[f'{int(rider)}'], matched_driver_location[f'{int(driver)}']],
                color='green', 
                weight=5,
                tooltip='matched_links'
                ).add_to(m)
    
        display(m)
        pass