from V1model import V1model
import torch
import torch.nn as nn
import torch.nn.functional as F


class classicalv1(nn.Module):
    def __init__(
        self,
        inputs_res=(64, 64),
        inputs_fov=(8, 8),
        cell_density=2,
        n_ori=8,
        n_phase=4,
        sf=[0.5, 1, 2],
        sigma_x_values=[0.6],
        sigma_y_values=[0.6],
        with_complex_cells=True,
    ):
        super().__init__()
        self.model = V1model(
            inputs_res,
            inputs_fov,
            cell_density,
            n_ori,
            n_phase,
            sf,
            sigma_x_values,
            sigma_y_values,
            with_complex_cells,
        )
        self.simple_cell_filters = nn.Parameter(
            torch.Tensor(self.model.get_simple_cell_filters()), requires_grad=False
        )
        (f1, f2) = self.model.get_complex_cell_filters()
        self.complex_cell_f1s = nn.Parameter(torch.Tensor(f1), requires_grad=False)
        self.complex_cell_f2s = nn.Parameter(torch.Tensor(f2), requires_grad=False)
        self.params = nn.ParameterDict(
            {
                "simple_cells": self.simple_cell_filters,
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
        xs = self.simple_cells_forward(x)
        xc = self.complex_cells_forward(x)
        resp = torch.cat((xs, xc), 1)
        return resp
