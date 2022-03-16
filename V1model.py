#%%
import numpy as np
from V1cells import *


class V1model:
    """
    Classical V1 model
    """

    def __init__(
        self,
        inputs_res,
        inputs_fov,
        n_pos_per_deg,
        n_ori,
        n_phase,
        sf,
        sigma_x_values=[1],
        sigma_y_values=[1],
        with_simple_cells=True,
        with_complex_cells=True,
        noisy_responses=False,
    ):
        """init function of a classical V1 model

        Args:
            inputs_res (int, int): dimension of the inputs and of the cells filters in x and y dimension
            inputs_fov (float, float): field of view in degree of visual field covered by the input
            n_pos_per_deg (float): number of cells per degree of visual field
            n_ori (int): number of equidistant orientations of cell's Gabor filters
            n_phase (int): number of equidistant phases of cell's Gabor filters
            sf (list of floats): spatial frequencies values of cell's Gabor filters
            sigma_x_values (list of floats, optional): standard deviation of the Gaussian envelope of the 
                                                    cell's Gabor filters in the orthogonal direction. Defaults to [1].
            sigma_y_values (list of floats, optional): standard deviation of the Gaussian envelope of the 
                                                    cell's Gabor filters in the parallel direction. Defaults to [1].
            with_complex_cells (bool, optional): True to have complex cells (on top of simples). Defaults to True.
        """
        self.inputs_res = inputs_res
        self.inputs_fov = inputs_fov
        self.xlim = (-inputs_fov[0] / 2, inputs_fov[0] / 2)
        self.ylim = (-inputs_fov[1] / 2, inputs_fov[1] / 2)
        self.n_pos_per_deg = n_pos_per_deg
        self.n_ori = n_ori
        self.sf = sf
        if type(self.sf) != list:
            self.sf = [self.sf]
        self.n_phase = n_phase
        self.sigma_x_values = sigma_x_values
        self.sigma_y_values = sigma_y_values
        self.positions = self._get_positions()
        self.orientations = self._get_orientations()
        self.phases = self._get_phases()
        self.with_simple_cells = with_simple_cells
        self.with_complex_cells = with_complex_cells
        if self.with_simple_cells:
            self.simple_cells = self.create_simple_cells()
        else:
            self.simple_cells = []
        if self.with_complex_cells == True:
            self.complex_cells = self.create_complex_cells()
        else:
            self.complex_cells = []
        self.cells = self.simple_cells + self.complex_cells
        self.noisy_responses = noisy_responses

    def _get_positions(self):
        pos_x = (
            np.linspace(-0.5, 0.5, int(self.n_pos_per_deg * self.inputs_fov[0]))
            * self.inputs_fov[0]
        )
        pos_y = (
            np.linspace(-0.5, 0.5, int(self.n_pos_per_deg * self.inputs_fov[1]))
            * self.inputs_fov[1]
        )
        pos_x = 1 / (self.n_pos_per_deg) + pos_x[:-1]
        pos_y = 1 / (self.n_pos_per_deg) + pos_y[:-1]
        positions = [(x, y) for x in pos_x for y in pos_y]
        return positions

    def _get_orientations(self):
        orientations = np.linspace(0, np.pi, self.n_ori + 1)[:-1]
        return orientations

    def _get_phases(self):
        phases = np.linspace(0, 2 * np.pi, self.n_phase + 1)[:-1]
        return phases

    def create_simple_cells(self):
        """creates the models simple cells according to the initialization args

        Returns:
            [list]: list containing simple cells objects 
        """
        return [
            simple_cell(
                self.inputs_res,
                self.xlim,
                self.ylim,
                pos,
                theta,
                sf,
                phase,
                sigma_x,
                sigma_y,
            )
            for pos in self.positions
            for theta in self.orientations
            for sf in self.sf
            for phase in self.phases
            for sigma_x in self.sigma_x_values
            for sigma_y in self.sigma_y_values
        ]

    def create_complex_cells(self):
        """creates the models complex cells according to the initialization args

        Returns:
            [list]: list containing complex cells objects 
        """
        return [
            complex_cell(
                self.inputs_res, self.xlim, self.ylim, pos, theta, sf, sigma_x, sigma_y,
            )
            for pos in self.positions
            for theta in self.orientations
            for sf in self.sf
            for sigma_x in self.sigma_x_values
            for sigma_y in self.sigma_y_values
        ]

    def get_responses(self, stim):
        """returns cells responses to a given stimulus

        Args:
            stim ([2d numpy array]): stimulus

        Returns:
            [list of floats]: list containing cell's responses
        """
        resp = [cell.get_response(stim) for cell in self.cells]
        if self.noisy_responses:
            resp = [np.random.poisson(r) for r in resp]
        return resp

    def get_simple_cell_filters(self):
        """function to get all the filters of simple cells as a 3d numpy array 

        Returns:
            [3d numpy array]: cell's filters
        """
        simple_cells_filters = np.array([cell.filter for cell in self.simple_cells])
        return simple_cells_filters

    def get_complex_cell_filters(self):
        """function to get the filters and the 180 deg phase shifted filters of complex cells

        Returns:
            [list]: list containing 2 3d numpy arrays representing the cells
                    filters and phase shifted filters
        """
        complex_cells_f1s = np.array([cell.f1 for cell in self.complex_cells])
        complex_cells_f2s = np.array([cell.f2 for cell in self.complex_cells])
        return complex_cells_f1s, complex_cells_f2s


class V1model_random:
    """
    Classical V1 model with randomly extracted property values
    """

    def __init__(
        self,
        inputs_res,
        inputs_fov,
        n_cells=10000,
        sf_values=[0.5, 0.7, 0.9, 1.1],
        sigma=[0.2, 0.3, 0.4, 0.5],
        gabor_aspect_ratio=[0.6, 0.8, 1],
        with_complex_cells=True,
        noisy_responses=False,
        simple_complex_ratio=0.5,
    ):

        self.inputs_res = inputs_res
        self.inputs_fov = inputs_fov
        self.n_cells = n_cells
        self.simple_complex_ratio = simple_complex_ratio
        if with_complex_cells:
            self.n_simple_cells = int(np.ceil(self.n_cells * simple_complex_ratio))
            self.n_complex_cells = int(self.n_cells - self.n_simple_cells)
        else:
            self.n_simple_cells = self.n_cells
            self.n_complex_cells = 0
        self.xlim = (-inputs_fov[0] / 2, inputs_fov[0] / 2)
        self.ylim = (-inputs_fov[1] / 2, inputs_fov[1] / 2)
        self.sf = sf_values
        if type(self.sf) != list:
            self.sf = [self.sf]

        self.sigma = sigma
        if type(self.sigma) != list:
            self.sigma = [self.sigma]
        self.gabor_aspect_ratio = gabor_aspect_ratio

        if type(self.gabor_aspect_ratio) != list:
            self.gabor_aspect_ratio = [self.gabor_aspect_ratio]
        self.noisy_responses = noisy_responses
        self.with_complex_cells = with_complex_cells

        self.simple_cells, self.simple_cell_props = self.create_simple_cells()
        if self.with_complex_cells == True:
            self.complex_cells, self.complex_cells_props = self.create_complex_cells()
        else:
            self.complex_cells = []
        self.cells = self.simple_cells + self.complex_cells

    def gen_properties(self, n, phase=True):
        properties = {}
        properties["pos"] = self.gen_positions(n)
        properties["ori"] = np.random.uniform(0, np.pi, size=n)
        if phase == True:
            properties["phase"] = np.random.uniform(0, 2 * np.pi, size=n)
        properties["sf"] = np.random.choice(self.sf, size=n)
        properties["sigma_x"] = np.random.choice(self.sigma, size=n)
        properties["sigma_y"] = properties["sigma_x"] / np.random.choice(
            self.gabor_aspect_ratio, size=n
        )
        return properties

    def gen_positions(self, n):
        # pos_x = np.random.uniform(low=-0.5, high=0.5, size=n) * self.inputs_fov[0]
        # pos_y = np.random.uniform(low=-0.5, high=0.5, size=n) * self.inputs_fov[1]
        positions = np.random.uniform(low=-0.5, high=0.5, size=[n, 2])
        positions[:, 0] = positions[:, 0] * self.inputs_fov[0]
        positions[:, 1] = positions[:, 1] * self.inputs_fov[1]
        # positions = [(x, y) for x in pos_x for y in pos_y]
        return positions

    def create_simple_cells(self):
        print("creating " + str(self.n_simple_cells) + " simple cells")
        props = self.gen_properties(self.n_simple_cells, phase=True)
        cells = [
            simple_cell(
                self.inputs_res,
                self.xlim,
                self.ylim,
                pos,
                theta,
                sf,
                phase,
                sigmax,
                sigmay,
            )
            for pos, theta, sf, phase, sigmax, sigmay in zip(
                props["pos"],
                props["ori"],
                props["sf"],
                props["phase"],
                props["sigma_x"],
                props["sigma_y"],
            )
        ]
        return cells, props

    def create_complex_cells(self):
        print("creating " + str(self.n_complex_cells) + " complex cells")
        props = self.gen_properties(self.n_complex_cells, phase=False)
        cells = [
            complex_cell(
                self.inputs_res, self.xlim, self.ylim, pos, theta, sf, sigmax, sigmay,
            )
            for pos, theta, sf, sigmax, sigmay in zip(
                props["pos"],
                props["ori"],
                props["sf"],
                props["sigma_x"],
                props["sigma_y"],
            )
        ]
        return cells, props

    def get_responses(self, stim):
        """returns cells responses to a given stimulus

        Args:
            stim ([2d numpy array]): stimulus

        Returns:
            [list of floats]: list containing cell's responses
        """
        resp = [cell.get_response(stim) for cell in self.cells]
        if self.noisy_responses:
            resp = [np.random.poisson(r) for r in resp]
        return resp

    def get_simple_cell_filters(self):
        """function to get all the filters of simple cells as a 3d numpy array 

        Returns:
            [3d numpy array]: cell's filters
        """
        simple_cells_filters = np.array([cell.filter for cell in self.simple_cells])
        return simple_cells_filters

    def get_complex_cell_filters(self):
        """function to get the filters and the 180 deg phase shifted filters of complex cells

        Returns:
            [list]: list containing 2 3d numpy arrays representing the cells
                    filters and phase shifted filters
        """
        complex_cells_f1s = np.array([cell.f1 for cell in self.complex_cells])
        complex_cells_f2s = np.array([cell.f2 for cell in self.complex_cells])
        return complex_cells_f1s, complex_cells_f2s

