import numpy as np
from GaborFilters import GaborFilter


class V1cell:
    """
    Parent class for V1 simple and complex cells 
    """

    def __init__(self, res, xlim, ylim, pos, theta, sf, phase, sigma_x, sigma_y):
        self.res = res
        self.xlim = xlim
        self.ylim = ylim
        self.pos = pos
        self.theta = theta
        self.sf = sf
        self.phase = phase
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y


class simple_cell(V1cell):
    def __init__(self, res, xlim, ylim, pos, theta, sf, phase, sigma_x, sigma_y):
        """V1 simple cell

        Args:
            res (list): list containing the dimensions of filter and stimuli in x and y directions
            xlim (list): list containing minimum and maximum value of x in degrees of visual field
            ylim (list): list containing minimum and maximum value of y in degrees of visual field
            pos (list): list containing x and y position of the center of the cell's Gabor filter
            theta (float): orientation of the cell's Gabor filter
            sf (float): spatial frequency of the cell's Gabor filter
            phase (float): phase of the cell's Gabor filter
            sigma_x (float): std of the Gaussian envelope of the cell's Gabor filter in the orthogonal diretion
            sigma_y (float): std of the Gaussian envelope of the cell's Gabor filter in the parallel diretion
        """
        super().__init__(
            res=res,
            xlim=xlim,
            ylim=ylim,
            pos=pos,
            theta=theta,
            sf=sf,
            phase=phase,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
        )
        self.type = "simple"
        self.filter = GaborFilter(
            pos=self.pos,
            sf=self.sf,
            phase=self.phase,
            theta=self.theta,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            res=self.res,
            xlim=self.xlim,
            ylim=self.ylim,
        )

    def get_response(self, stim):
        x = np.multiply(self.filter, stim).sum()
        resp = x * (x > 0)
        return resp


class complex_cell(V1cell):
    def __init__(self, res, xlim, ylim, pos, theta, sf, sigma_x, sigma_y):

        """V1 complex cell based on the energy model

        Args:
            res (list): list containing the dimensions of filter and stimuli in x and y directions
            xlim (list): list containing minimum and maximum value of x in degrees of visual field
            ylim (list): list containing minimum and maximum value of y in degrees of visual field
            pos (list): list containing x and y position of the center of the cell's Gabor filter
            theta (float): orientation of the cell's Gabor filter
            sf (float): spatial frequency of the cell's Gabor filter
            sigma_x (float): std of the Gaussian envelope of the cell's Gabor filter in the orthogonal diretion
            sigma_y (float): std of the Gaussian envelope of the cell's Gabor filter in the parallel diretion
        """
        super().__init__(
            res=res,
            xlim=xlim,
            ylim=ylim,
            pos=pos,
            theta=theta,
            sf=sf,
            phase=None,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
        )
        self.type = "complex"

        self.f1 = GaborFilter(
            pos=self.pos,
            sf=self.sf,
            phase=0,
            theta=self.theta,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            res=self.res,
            xlim=self.xlim,
            ylim=self.ylim,
        )
        self.f2 = GaborFilter(
            pos=self.pos,
            sf=self.sf,
            phase=np.pi / 2,
            theta=self.theta,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            res=self.res,
            xlim=self.xlim,
            ylim=self.ylim,
        )

    def get_response(self, stim):
        x1 = np.multiply(self.f1, stim).sum()
        x2 = np.multiply(self.f2, stim).sum()
        resp = np.sqrt(x1 ** 2 + x2 ** 2)
        return resp

