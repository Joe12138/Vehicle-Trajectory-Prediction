import logging

import numpy as np


class RoadNetwork(object):
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        """
        A node represents an symbolic intersection in the road network.
        :param node: the node label.
        """
        if node not in self.graph:
            self.graph[node] = []

    def add_lane(self, _from, _to, lane):
        """
        A lane is encoded as an edge in the road network.
        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index):
        """
        Get the lane geometry corresponding to a given index in the road network.
        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry
        """
        _from, _to, _id = index
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(self, position):
        """
        Get the index of the lane closest to a world postion.
        
        :param position: a world position [m]
        :return:  the index of the clsest lane.
        """

        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance(position))
                    indexes.append((_from, _to, _id))

        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index, route=None, position=None, np_random=np.random):
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        If a plan is available and matches with current lane, follow it.
        Else, pick next road randomly.
        if it has the same number of lanes as current road, stay in the same lane.
        Else, pick next road's closest lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.

        :return: the index of the next lane to the followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = None

        # Pick next road according to planned route
        if route:
            if route[0][:2] == current_index[:2]:  # we just finished the first step of the route, drop it.
                route.pop(0)
            if route and route[0][0] == _to:  # Next road in route is starting at the end of current road.
                _, next_to, route_id = route[0]
            elif route:
                logging.warning("Route {} does not start after current road {}.".format(route[0], current_index))
        # Randomly pick next road
        if not next_to:
            try:
                next_to = list(self.graph[_to].keys())[np_random.randint(len(self.graph[_to]))]
            except KeyError:
                return current_index

        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes, key=lambda l: self.get_lane((_to, next_to, l))).distance(position)

        return _to, next_to, next_id

    def bfs_paths(self, start, goal):
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal
        """
        queue = [(start, [start])]

        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []

            for _next in set(self.graph[node].keys())-set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path+[_next]))

    def shortest_path(self, start, goal):
        """
        Breadth-first search of shortest path from start to goal.
        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal
        """
        try:
            return next(self.bfs_paths(start, goal))
        except StopIteration:
            return None

    def side_lane(self, lane_index):
        """
        :param lane_index: the index of a lane
        :return: indexes of lanes next to an input lane, to its right or left or both.
        """
        _from, _to, _id = lane_index
        lanes = []

        if _id > 0:
            lanes.append((_from, _to, _id-1))
        if _id < len(self.graph[_from][_to])-1:
            lanes.append((_from, _to, _id+1))
        return lanes