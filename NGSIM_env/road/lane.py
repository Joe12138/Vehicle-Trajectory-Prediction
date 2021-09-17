from abc import ABCMeta, abstractclassmethod
import numpy as np
from NGSIM_env.vehicle.dynamics import Vehicle

class AbstratcLane(object):
    """
    A lane on the road, described by its central curve
    """
    metaclass__ = ABCMeta
    DEFAULT_WIDTH = 3

    @abstractclassmethod
    def position(self, longitudinal, lateral):
        """
        Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding world position [m]
        """

        raise NotImplementedError()

    @abstractclassmethod
    def local_coordinates(self, position):
        """
        Convert a workd position to local lane coodinates.

        :param position: a world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()


    @abstractclassmethod
    def headding_at(self, longitudinal):
        """
        Get the lane heading at a given longitudinal lane coordinate.
        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractclassmethod
    def width_at(self, longitudinal):
        """
        Get the lane width at a given longitudinal lane coodinate.
        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    def on_lane(self, position, longitudinal=None, lateral=None, margin=0):
        """
        whether a given world position is on the lane,

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if know [m]
        :param lateral: (optional) the corrsponding lateral lane coordinate, if know [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if not longitudinal or not lateral:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = np.abs(lateral) <= self.width_at(longitudinal)/2+margin and -Vehicle.LENGTH <= longitudinal < self.length+Vehicle.LENGTH

        return is_on

    def is_reachable_from(self, position):
        """
        Whether the lane is reachable from a given world position
        
        :param position: the world position [m]
        :return: is the lane reachable ?
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = np.abs(lateral) <= 2 * self.width_at(longitudinal) and 0 <= longitudinal < self.length + Vehicle.LENGTH
        return is_close

    def distance(self, position):
        """
        Compute the L1 distance [m] from a position to the lane
        """
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)


class LineType:
    """
    A lane side line type.
    """

    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


class StraightLane(AbstratcLane):
    """
    A lane going in straight line
    """

    def __init__(self, start, end, width=AbstratcLane.DEFAULT_WIDTH, line_types=None, forbidden=False, speed_limit=20, priority=0):
        """
        New straight lane

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param width: the lane width [m]
        :param line_types: the type of lines on both sides of the lane
        :param forbidden: is changing to this lane forbidden
        :param priority: priority level of the lane, for derermining who has right of way
        """
        super(StraightLane, self).__init__()
        self.start = np.array(start)
        self.end = np.array(end)
        self.width = width
        # 道路朝向的角度
        self.headding = np.arctan2(self.end[1]-self.start[1], self.end[0]-self.start[0])
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end-self.start)/self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden
        self.prioriy = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal, lateral):
        return self.start+longitudinal*self.direction+lateral*self.direction_lateral

    def headding_at(self, longitudinal):
        return self.headding

    def width_at(self, longitudinal):
        return self.width

    def local_coordinates(self, position):
        delta = position-self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)

        return longitudinal, lateral

