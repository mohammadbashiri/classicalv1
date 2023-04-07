from classicalv1.V1model import V1model, V1model_random
import torch
import torch.nn as nn
import torch.nn.functional as F


class classicalV1(nn.Module):
    def __init__(
        self,
        inputs_res=(64, 64),
        inputs_fov=(8, 8),
        n_pos_per_deg=2,
        n_ori=8,
        n_phase=4,
        sf=[0.5, 1, 2],
        sigma_x_values=[0.6],
        sigma_y_values=[0.6],
        with_simple_cells=True,
        with_complex_cells=True,
        noisy_responses=False,
        subset_of_cells_dict=None,
    ):
        super().__init__()
        self.model = V1model(
            inputs_res,
            inputs_fov,
            n_pos_per_deg,
            n_ori,
            n_phase,
            sf,
            sigma_x_values,
            sigma_y_values,
            with_simple_cells,
            with_complex_cells,
            noisy_responses,
        )
        params = {}
        if self.model.with_simple_cells == True:
            self.simple_cell_filters = nn.Parameter(
                torch.Tensor(self.model.get_simple_cell_filters()), requires_grad=False
            )
            params.update({"simple_cells": self.simple_cell_filters})
        if self.model.with_complex_cells == True:
            (f1, f2) = self.model.get_complex_cell_filters()
            self.complex_cell_f1s = nn.Parameter(torch.Tensor(f1), requires_grad=False)
            self.complex_cell_f2s = nn.Parameter(torch.Tensor(f2), requires_grad=False)
            params.update(
                {
                    "complex_cells_f1": self.complex_cell_f1s,
                    "complex_cells_f2": self.complex_cell_f2s,
                }
            )
        self.params = nn.ParameterDict(params)
        if subset_of_cells_dict != None:
            self.keep_subset_of_cells(**subset_of_cells_dict)
        self.eval()

    def keep_subset_of_cells(self, simple_cells_idx=None, complex_cells_idx=None):
        if simple_cells_idx != None:
            self.model.simple_cells = [
                self.model.simple_cells[i] for i in simple_cells_idx
            ]
        if complex_cells_idx != None:
            self.model.complex_cells = [
                self.model.complex_cells[i] for i in complex_cells_idx
            ]
        if self.model.with_simple_cells == True:
            self.simple_cell_filters = nn.Parameter(
                torch.Tensor(self.model.get_simple_cell_filters()), requires_grad=False
            )
            self.params.update({"simple_cells": self.simple_cell_filters})
        if self.model.with_complex_cells == True:
            (f1, f2) = self.model.get_complex_cell_filters()
            self.complex_cell_f1s = nn.Parameter(torch.Tensor(f1), requires_grad=False)
            self.complex_cell_f2s = nn.Parameter(torch.Tensor(f2), requires_grad=False)
            self.params.update(
                {
                    "complex_cells_f1": self.complex_cell_f1s,
                    "complex_cells_f2": self.complex_cell_f2s,
                }
            )

    def simple_cells_forward(self, x):
        x = torch.tensordot(x, self.simple_cell_filters, dims=[[1, 2], [1, 2]])
        x = F.relu(x)
        return x

    def complex_cells_forward(self, x):
        x1 = torch.tensordot(x, self.complex_cell_f1s, dims=[[1, 2], [1, 2]])
        x2 = torch.tensordot(x, self.complex_cell_f2s, dims=[[1, 2], [1, 2]])
        xc = torch.sqrt(x1 ** 2 + x2 ** 2)
        return xc

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        assert len(x.shape) == 3
        resp = []
        if self.model.with_simple_cells == True:
            resp.append(self.simple_cells_forward(x))
        if self.model.with_complex_cells == True:
            resp.append(self.complex_cells_forward(x))
        resp = torch.cat(resp, 1)
        return resp


class classicalV1random(nn.Module):
    def __init__(
        self,
        inputs_res=(64, 64),
        inputs_fov=(11, 11),
        n_cells=10000,
        sf_values=[0.5, 0.7, 0.9, 1.1],
        sigma=[0.2, 0.3, 0.4, 0.5],
        gabor_aspect_ratio=[0.4, 0.6, 0.8],
        with_complex_cells=True,
        noisy_responses=False,
        simple_complex_ratio=0.5,
    ):
        super().__init__()
        self.model = V1model_random(
            inputs_res=inputs_res,
            inputs_fov=inputs_fov,
            n_cells=n_cells,
            sf_values=sf_values,
            sigma=sigma,
            gabor_aspect_ratio=gabor_aspect_ratio,
            with_complex_cells=with_complex_cells,
            noisy_responses=noisy_responses,
            simple_complex_ratio=simple_complex_ratio,
        )
        self.simple_cell_filters = nn.Parameter(
            torch.Tensor(self.model.get_simple_cell_filters()), requires_grad=False
        )
        params = {"simple_cells": self.simple_cell_filters}
        if self.model.with_complex_cells == True:
            (f1, f2) = self.model.get_complex_cell_filters()
            self.complex_cell_f1s = nn.Parameter(torch.Tensor(f1), requires_grad=False)
            self.complex_cell_f2s = nn.Parameter(torch.Tensor(f2), requires_grad=False)
            params.update(
                {
                    "complex_cells_f1": self.complex_cell_f1s,
                    "complex_cells_f2": self.complex_cell_f2s,
                }
            )
        self.params = nn.ParameterDict(params)

    def simple_cells_forward(self, x):
        x = torch.tensordot(x, self.simple_cell_filters, dims=[[1, 2], [1, 2]])
        x = F.relu(x)
        return x

    def complex_cells_forward(self, x):
        x1 = torch.tensordot(x, self.complex_cell_f1s, dims=[[1, 2], [1, 2]])
        x2 = torch.tensordot(x, self.complex_cell_f2s, dims=[[1, 2], [1, 2]])
        xc = torch.sqrt(x1 ** 2 + x2 ** 2)
        return xc

    def forward(self, x):
        with torch.no_grad():
            if self.model.with_complex_cells == True:
                xs = self.simple_cells_forward(x)
                xc = self.complex_cells_forward(x)
                resp = torch.cat((xs, xc), 1)
            else:
                resp = self.simple_cells_forward(x)
        return resp
