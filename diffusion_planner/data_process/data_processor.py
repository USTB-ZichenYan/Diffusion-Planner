import numpy as np
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
agent_past_process, 
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list,
agent_future_process
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs


class DataProcessor(object):
    def __init__(self, config):
        """
        初始化数据处理器对象。

        参数:
            config (object): 配置对象，包含各种配置参数。

        返回:
            无
        """
        self._save_dir = getattr(config, "save_path", None) 

        self.past_time_horizon = 2  # 定义过去时间范围为2秒
        self.num_past_poses = 10 * self.past_time_horizon  # 计算过去姿势的数量
        self.future_time_horizon = 8  # 定义未来时间范围为8秒
        self.num_future_poses = 10 * self.future_time_horizon  # 计算未来姿势的数量

        self.num_agents = config.agent_num  # 设置代理数量
        self.num_static = config.static_objects_num  # 设置静态物体数量
        self.max_ped_bike = 10  # 限制代理中的行人和自行车数量
        self._radius = 100  # 查询半径范围，单位为米

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES']  # 要提取的地图特征名称
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num}  # 每个特征层要提取的最大元素数
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len}  # 每个特征层要提取的每个特征的最大点数

    # Use for inference
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_agent_past = None # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

        
        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": np.array([0., 0., 1. ,0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), # ego centric x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate, we only use x, y, cos, sin during inference
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data
    
    # Use for data preprocess
    def work(self, scenarios):
        """
        处理场景数据并生成训练所需的格式。

        参数:
            scenarios (List[Scenario]): 要处理的场景列表，每个场景包含环境和智能体的历史状态信息。

        返回:
            无：处理结果直接保存到磁盘上。
        """
        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api        

            '''
            ego & agents past
            '''
            # 提取自车初始状态以及历史轨迹
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses, self.past_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            # 获取过去时间范围内所有跟踪对象的数据
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]
            # 将采样得到的观测转换为数组形式
            neighbor_agents_past, neighbor_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)
            
            # 提取静态物体数据
            static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

            # 对代理和静态物体的历史进行处理以获得固定大小的输出
            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
            
            '''
            Map
            '''
            # 获取路线上的道路块ID及交通灯状态
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
            print("route_roadblock_ids: ", route_roadblock_ids)
            print("traffic_light_data: ", traffic_light_data)
            '''
            route_roadblock_ids:  
            ['19334', '19082', '19710', '19190', '19658', '19208', '19652', '19207', 
            '19207', '19187', '19678', '19182', '19683', '19165', '19616', '19164', '19622']
            traffic_light_data:  
            [TrafficLightStatusData(status=<TrafficLightStatusType.RED: 2>, lane_connector_id=18564, timestamp=1629227774200103), 
            TrafficLightStatusData(status=<TrafficLightStatusType.RED: 2>, lane_connector_id=19950, timestamp=1629227774200103), 
            TrafficLightStatusData(status=<TrafficLightStatusType.RED: 2>, lane_connector_id=20370, timestamp=1629227774200103), 
            TrafficLightStatusData(status=<TrafficLightStatusType.RED: 2>, lane_connector_id=18871, timestamp=1629227774200103)]
            '''

            # 校正路线上的道路块ID
            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids
                )
            print("route_roadblock_ids_post: ", route_roadblock_ids)

            # 获取邻近地图元素及其特征
            coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
                map_api, self._map_features, ego_coords, self._radius, traffic_light_data
            )

            # 处理地图信息以适应模型输入要求
            vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

            '''
            ego & agents future
            '''
            # 获取自车未来轨迹
            ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)

            # 获取未来时间范围内所有跟踪对象的数据
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]

            sampled_future_observations = [present_tracked_objects] + future_tracked_objects
            future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
            # 处理邻居代理未来的轨迹信息
            neighbor_agents_future = agent_future_process(anchor_ego_state, future_tracked_objects_array_list, self.num_agents, neighbor_indices)


            '''
            ego current
            '''
            # 计算额外的自车当前状态
            ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

            # gather data
            data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
            data.update(vector_map)

            self.save_to_disk(self._save_dir, data)

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)