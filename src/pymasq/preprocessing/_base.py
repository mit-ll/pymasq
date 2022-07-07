from abc import abstractmethod


# This is the base class for all preprocessors


class PreprocessorBase:
    def __init__(self):
        pass

    @abstractmethod
    def encode(self):
        """
        take a data frame and encode the values of all columns
        """
        pass

    @abstractmethod
    def encode_both(self):
        """
        take two data frames and ensure that the encoding they use is consistent.
        For example, if one df has values [A,B] and the other has [B,C], they might be
        encoded as [0,1] and [1,2]. In contract, two distinct calls to encode() will
        return [0,1] and [0,1]
        """
        pass