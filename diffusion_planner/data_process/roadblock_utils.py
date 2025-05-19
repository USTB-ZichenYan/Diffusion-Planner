import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Union, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject


def normalize_angle(angle: np.ndarray):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class BreadthFirstSearchRoadBlock:
    """
    A class that performs iterative breadth first search. The class operates on the roadblock graph.
    """

    def __init__(
        self, start_roadblock_id: int, map_api: Optional[AbstractMap], forward_search: str = True
    ):
        """
        Constructor of BreadthFirstSearchRoadBlock class
        :param start_roadblock_id: roadblock id where graph starts
        :param map_api: map class in nuPlan
        :param forward_search: whether to search in driving direction, defaults to True
        """
        self._map_api: Optional[AbstractMap] = map_api
        self._queue = deque([self.id_to_roadblock(start_roadblock_id), None])
        self._parent: Dict[str, Optional[RoadBlockGraphEdgeMapObject]] = dict()
        self._forward_search = forward_search

        #  lazy loaded
        self._target_roadblock_ids: List[str] = None

    def search(
        self, target_roadblock_id: Union[str, List[str]], max_depth: int
    ) -> Tuple[List[RoadBlockGraphEdgeMapObject], bool]:
        """
        Apply BFS to find route to target roadblock.
        :param target_roadblock_id: id of target roadblock
        :param max_depth: maximum search depth
        :return: tuple of route and whether a path was found
        """

        if isinstance(target_roadblock_id, str):
            target_roadblock_id = [target_roadblock_id]
        self._target_roadblock_ids = target_roadblock_id

        start_edge = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_edge: RoadBlockGraphEdgeMapObject = start_edge
        end_depth: int = 1
        depth: int = 1

        self._parent[start_edge.id + f"_{depth}"] = None

        while self._queue:
            current_edge = self._queue.popleft()

            # Early exit condition
            if self._check_end_condition(depth, max_depth):
                break

            # Depth tracking
            if current_edge is None:
                depth += 1
                self._queue.append(None)
                if self._queue[0] is None:
                    break
                continue

            # Goal condition
            if self._check_goal_condition(current_edge, depth, max_depth):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                break

            neighbors = (
                current_edge.outgoing_edges if self._forward_search else current_edge.incoming_edges
            )

            # Populate queue
            for next_edge in neighbors:
                # if next_edge.id in self._candidate_lane_edge_ids_old:
                self._queue.append(next_edge)
                self._parent[next_edge.id + f"_{depth + 1}"] = current_edge
                end_edge = next_edge
                end_depth = depth + 1

        return self._construct_path(end_edge, end_depth), path_found

    def id_to_roadblock(self, id: str) -> RoadBlockGraphEdgeMapObject:
        """
        Retrieves roadblock from map-api based on id
        :param id: id of roadblock
        :return: roadblock class
        """
        block = self._map_api._get_roadblock(id)
        block = block or self._map_api._get_roadblock_connector(id)
        return block

    @staticmethod
    def _check_end_condition(depth: int, max_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: whether depth exceeds the target depth.
        """
        return depth > max_depth

    def _check_goal_condition(
        self,
        current_edge: RoadBlockGraphEdgeMapObject,
        depth: int,
        max_depth: int,
    ) -> bool:
        """
        Check if the current edge is at the target roadblock at the given depth.
        :param current_edge: edge to check.
        :param depth: current depth to check.
        :param max_depth: maximum depth the edge should be at.
        :return: True if the lane edge is contain the in the target roadblock. False, otherwise.
        """
        return current_edge.id in self._target_roadblock_ids and depth <= max_depth

    def _construct_path(
        self, end_edge: RoadBlockGraphEdgeMapObject, depth: int
    ) -> List[RoadBlockGraphEdgeMapObject]:
        """
        Constructs a path when goal was found.
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of RoadBlockGraphEdgeMapObject
        """
        path = [end_edge]
        path_id = [end_edge.id]

        while self._parent[end_edge.id + f"_{depth}"] is not None:
            path.append(self._parent[end_edge.id + f"_{depth}"])
            path_id.append(path[-1].id)
            end_edge = self._parent[end_edge.id + f"_{depth}"]
            depth -= 1

        if self._forward_search:
            path.reverse()
            path_id.reverse()

        return (path, path_id)


def get_current_roadblock_candidates(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblocks_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    heading_error_thresh: float = np.pi / 4,
    displacement_error_thresh: float = 3,
) -> Tuple[RoadBlockGraphEdgeMapObject, List[RoadBlockGraphEdgeMapObject]]:
    """
    该函数用于确定当前自车（ego vehicle）所在的道路块（roadblock）。它通过以下步骤实现这一目标：

    1. **查找附近的 Roadblock**：
    - 首先，函数会查找距离自车当前位置较近的 roadblock 和 roadblock connector。
    - 如果没有找到任何附近的 roadblock，则进一步搜索最近的 roadblock。

    2. **评估候选 Roadblock**：
    - 对于每个候选的 roadblock，函数会检查其内部的车道（interior edges）。
    - 在每个车道上，计算自车位置与车道路径点之间的位移误差和航向角误差。
    - 如果误差在设定的阈值范围内，则认为该 roadblock 是一个有效的候选。

    3. **优先选择规划路线上的 Roadblock**：
    - 函数优先选择位于规划路线上的 roadblock，并根据位移误差选择最优匹配。
    - 如果没有符合要求的 on-route roadblock，则回退到最合适的 off-route 候选。

    4. **返回结果**：
    - 返回最匹配的 roadblock 以及所有候选的 roadblock 列表。

    总的来说，该函数帮助自动驾驶系统识别当前车辆所在的位置，并为后续的路径规划提供依据。

    Determines a set of roadblock candidates where ego is located.

    This function identifies potential roadblocks near the ego vehicle's position. It prioritizes roadblocks that are on the planned route,
    and selects the most promising candidate based on heading and displacement errors.

    :param ego_state: Class containing the current state of the ego vehicle.
    :param map_api: Interface for accessing map data.
    :param route_roadblocks_dict: Dictionary of roadblocks that are on the planned route.
    :param heading_error_thresh: Maximum allowable heading error (in radians), defaults to π/4.
    :param displacement_error_thresh: Maximum allowable displacement error (in meters), defaults to 3.
    :return: A tuple containing the most promising roadblock and a list of other candidate roadblocks.
    """
    ego_pose: StateSE2 = ego_state.rear_axle
    roadblock_candidates = []

    # Get nearby roadblocks and roadblock connectors from the map
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    roadblock_dict = map_api.get_proximal_map_objects(
        point=ego_pose.point, radius=1.0, layers=layers
    )
    roadblock_candidates = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK]
        + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )
    print("roadblock_candidates:", len(roadblock_candidates), roadblock_candidates)

    # If no nearby roadblocks found, search for the nearest ones
    if not roadblock_candidates:
        for layer in layers:
            roadblock_id_, distance = map_api.get_distance_to_nearest_map_object(
                point=ego_pose.point, layer=layer
            )
            roadblock = map_api.get_map_object(roadblock_id_, layer)

            if roadblock:
                roadblock_candidates.append(roadblock)

    # Initialize lists to store on-route and off-route candidates along with their errors
    on_route_candidates, on_route_candidate_displacement_errors = [], []
    candidates, candidate_displacement_errors = [], []

    roadblock_displacement_errors = []
    roadblock_heading_errors = []

    # Evaluate each roadblock candidate to determine its suitability
    for idx, roadblock in enumerate(roadblock_candidates):
        lane_displacement_error, lane_heading_error = np.inf, np.inf
        print("roadblock: ", roadblock)
        print("roadblock.interior_edges: ", roadblock.interior_edges)
        print("ego_pose.point.array[None, ...]: ", ego_pose, ego_pose.point.array[None, ...])
        # Analyze each interior edge (lane) of the roadblock
        '''
        roadblock.interior_edges是roadblock的所有车道
        '''
        for lane in roadblock.interior_edges:
            lane_discrete_path: List[StateSE2] = lane.baseline_path.discrete_path
            lane_discrete_points = np.array(
                [state.point.array for state in lane_discrete_path], dtype=np.float64
            )
            print("lane_discrete_path: ", len(lane_discrete_path))
            print("lane_discrete_points: ", lane_discrete_points.shape)
            print("(lane_discrete_points - ego_pose.point.array[None, ...]) ** 2.0: ", (lane_discrete_points - ego_pose.point.array[None, ...]).shape)
            lane_state_distances = (
                (lane_discrete_points - ego_pose.point.array[None, ...]) ** 2.0
            ).sum(axis=-1) ** 0.5
            print("lane_state_distances: ", lane_state_distances.shape)
            argmin = np.argmin(lane_state_distances)

            heading_error = np.abs(
                normalize_angle(lane_discrete_path[argmin].heading - ego_pose.heading)
            )
            displacement_error = lane_state_distances[argmin]

            # Update minimum errors if a better match is found
            '''
            heading_error：自车航向与车道航向的角度差（归一化后绝对值，弧度）。
            heading_error_thresh：航向角误差阈值（默认π/4，约 45°），表示自车允许的最大航向偏差。
            displacement_error：自车位置到车道中心线的距离（米）。
            displacement_error_thresh：位移误差阈值（默认 3 米），表示自车允许的最大偏移。
            '''
            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            # Check if this lane meets the criteria for being a valid candidate
            if (
                heading_error < heading_error_thresh
                and displacement_error < displacement_error_thresh
            ):
                if roadblock.id in route_roadblocks_dict.keys():
                    on_route_candidates.append(roadblock)
                    on_route_candidate_displacement_errors.append(displacement_error)
                else:
                    candidates.append(roadblock)
                    candidate_displacement_errors.append(displacement_error)

        # Record the best errors for this roadblock
        roadblock_displacement_errors.append(lane_displacement_error)
        roadblock_heading_errors.append(lane_heading_error)

    # Prioritize on-route roadblocks
    if on_route_candidates:  # prefer on-route roadblocks
        return (
            on_route_candidates[np.argmin(on_route_candidate_displacement_errors)], #np.argmin(...)：找到误差最小的候选索引，对应的候选即为 “promising candidate”。
            on_route_candidates,
        )
    elif candidates:  # fallback to most promising candidate
        return candidates[np.argmin(candidate_displacement_errors)], candidates

    # Otherwise, just find any close roadblock
    return (
        roadblock_candidates[np.argmin(roadblock_displacement_errors)],
        roadblock_candidates,
    )


def route_roadblock_correction(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblock_ids: List[str],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> List[str]:
    """
    Applies several methods to correct route roadblocks.

    This function performs three main correction operations:
    1. Handles off-route starting positions
    2. Ensures connectivity between consecutive roadblocks
    3. Removes loops in the route path

    起点不在路线上：当车辆当前位置对应的道路块不在原规划路线中时，重新连接路线。
    道路块不连续：确保相邻道路块之间存在有效连接（如路口与道路的连通性）。
    路线环路：移除路线中可能存在的环路，避免车辆绕圈。

    ego_state：车辆当前状态（位置、朝向等）
    map_api：地图接口，用于获取道路块信息
    route_roadblock_ids：原规划路线的道路块 ID 列表
    search_depth_backward/forward：BFS 搜索的深度限制

    :param ego_state: class containing ego state
    :param map_api: map object providing access to map data
    :param route_roadblock_ids: list of roadblock IDs defining the planned route
    :param search_depth_backward: depth of backward BFS search for route correction, defaults to 15
    :param search_depth_forward: depth of forward BFS search for route correction, defaults to 30
    :return: list of corrected roadblock IDs forming a continuous valid route
    """

    route_roadblock_dict = {}
    for id_ in route_roadblock_ids:
        print("id: ", id_)
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        print("block: ", block)
        block = block or map_api.get_map_object(
            id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        print("block: ", block)
        route_roadblock_dict[id_] = block
    print("route_roadblock_dict: ", route_roadblock_dict)
    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_state, map_api, route_roadblock_dict
    )
    starting_block_ids = [roadblock.id for roadblock in starting_block_candidates]

    route_roadblocks = list(route_roadblock_dict.values())
    route_roadblock_ids = list(route_roadblock_dict.keys())

    # Fix 1: when agent starts off-route
    # Handle case where ego vehicle starts outside the defined route
    if starting_block.id not in route_roadblock_ids:
        # Backward search if current roadblock not in route
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )
        (path, path_id), path_found = graph_search.search(
            starting_block_ids, max_depth=search_depth_backward
        )

        if path_found:
            route_roadblocks[:0] = path[:-1]
            route_roadblock_ids[:0] = path_id[:-1]

        else:
            # Forward search to any route roadblock
            graph_search = BreadthFirstSearchRoadBlock(
                starting_block.id, map_api, forward_search=True
            )
            (path, path_id), path_found = graph_search.search(
                route_roadblock_ids[:3], max_depth=search_depth_forward
            )

            if path_found:
                end_roadblock_idx = np.argmax(
                    np.array(route_roadblock_ids) == path_id[-1]
                )

                route_roadblocks = route_roadblocks[end_roadblock_idx + 1 :]
                route_roadblock_ids = route_roadblock_ids[end_roadblock_idx + 1 :]

                route_roadblocks[:0] = path
                route_roadblock_ids[:0] = path_id

    # Fix 2: check if roadblocks are linked, search for links if not
    # Ensure connectivity between consecutive roadblocks in the route
    roadblocks_to_append = {}
    for i in range(len(route_roadblocks) - 1):
        next_incoming_block_ids = [
            _roadblock.id for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[i], map_api, forward_search=True
        )
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward
        )

        if path_found and path and len(path) >= 3:
            path, path_id = path[1:-1], path_id[1:-1]
            roadblocks_to_append[i] = (path, path_id)

    # append missing intermediate roadblocks
    offset = 1
    for i, (path, path_id) in roadblocks_to_append.items():
        route_roadblocks[i + offset : i + offset] = path
        route_roadblock_ids[i + offset : i + offset] = path_id
        offset += len(path)

    # Fix 3: cut route-loops
    # Remove any loops in the route path to ensure a clean trajectory
    route_roadblocks, route_roadblock_ids = remove_route_loops(
        route_roadblocks, route_roadblock_ids
    )

    return route_roadblock_ids


def remove_route_loops(
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
) -> Tuple[List[str], List[RoadBlockGraphEdgeMapObject]]:
    """
    Remove ending of route, if the roadblock are intersecting the route (forming a loop).
    :param route_roadblocks: input route roadblocks
    :param route_roadblock_ids: input route roadblocks ids
    :return: tuple of ids and roadblocks of route without loops
    """

    roadblock_occupancy_map = None
    loop_idx = None

    for idx, roadblock in enumerate(route_roadblocks):
        # loops only occur at intersection, thus searching for roadblock-connectors.
        if str(roadblock.__class__.__name__) == "NuPlanRoadBlockConnector":
            if not roadblock_occupancy_map:
                roadblock_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry(
                    [roadblock.polygon], [roadblock.id]
                )
                continue

            strtree, index_by_id = roadblock_occupancy_map._build_strtree()
            indices = strtree.query(roadblock.polygon)
            if len(indices) > 0:
                for geom in strtree.geometries.take(indices):
                    area = geom.intersection(roadblock.polygon).area
                    if area > 1:
                        loop_idx = idx
                        break
                if loop_idx:
                    break

            roadblock_occupancy_map.insert(roadblock.id, roadblock.polygon)

    if loop_idx:
        route_roadblocks = route_roadblocks[:loop_idx]
        route_roadblock_ids = route_roadblock_ids[:loop_idx]

    return route_roadblocks, route_roadblock_ids
