class Loggable(object):
    """
    Implements an object whose metrics can be logged through time and 
    accessed as a pandas Dataframe
    """

    def dump(self):
        """
        Update an internal log of object data.
        """
        raise Exception("Not implemented.")

    def get_log(self):
        """
        Case the object's internal log into a pandas Dataframe.

        @return: the DataFrame containing the object's log
        """

        raise Exception("Not implemented.")