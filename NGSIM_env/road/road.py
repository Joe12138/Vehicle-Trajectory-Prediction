import numpy as np


class RoadNetwork(object):
    def __init__(self):
        self.graph = {}

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
        for _from, to_dict in  self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance(position))
                    indexes.append((_from, _to, _id))

        return indexes[int(np.argmin(distances))]

    def bfs_path(self, start, goal):
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
            return next(self.bfs_path(start, goal))
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