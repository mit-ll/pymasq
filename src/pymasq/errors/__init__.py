
"""
Expose public exceptions & warnings
"""

class InputError(Exception):
    """ Exception raised for errors in the input value. """


class DataTypeError(Exception):
    """ Exception raised for errors in the data type. """
    
    
class SumNotEqualToOneError(ValueError):
    """ Exception for sum of values not equal to 1. """
    

class NotInRangeError(ValueError):
    """ Exception for values not in specified interval. """


class LessThanZeroError(ValueError):
    """ Exceptions for values < 0. """


class LessThanOrEqualToZeroError(ValueError):
    """ Exceptions for values <= 0. """


class NoMutationAvailableError(ValueError):
    """ Exception when all mutations have been discarded and not replaced """


__all__ = [
    "InputError",
    "DataTypeError",
    "SumNotEqualToOneError",
    "NotInRangeError",
    "LessThanZeroError",
    "LessThanOrEqualToZeroError",
    "NoMutationAvailableError"
]