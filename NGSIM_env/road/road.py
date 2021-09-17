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