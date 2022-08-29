import numpy as np


def GaborFilter(pos, sigma_x, sigma_y, sf, phase, theta, res, xlim, ylim):
    """
    Gabor function centered in pos=(pos_x, pos_y)
    pos (list): list containing x and y coordinate of the center of the filter 
    sigma_x (float): std of the gaussian envelop in the direction orthogonal to the grating
    sigma_y (float): std of the gaussian envelop in the direction parallel to the grating
    f (float): spatial frequency
    theta (float): orientation
    phi (float): angle
    xlim, ylim (lists): lists containing inferior and superior border of the filter in x and y axis
    res (list): list of dimension 2 containing resolution in x and y direction

    return: 2d matrix corresponding to the Gabor filter
    """
    pos_x, pos_y = pos
    x = np.linspace(xlim[0], xlim[1], res[0])
    y = np.linspace(ylim[0], ylim[1], res[1])
    X, Y = np.meshgrid(x, y)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    A = np.exp(
        -0.5
        * (
            ((X_rot + pos_x) ** 2 / sigma_x ** 2)
            + ((Y_rot + pos_y) ** 2 / sigma_y ** 2)
        )
    )
    B = np.cos(2 * np.pi * (X_rot + pos_x) * sf + phase)
    return A * B
