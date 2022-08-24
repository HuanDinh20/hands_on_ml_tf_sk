import numpy as np


def get_x_y():
    """This method returns random value shape (100, 1), it will  reused many time in this chapter"""
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y
