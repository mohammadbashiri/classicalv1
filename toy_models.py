#%%
from classicalv1.V1cells import complex_cell
from classicalv1.torchV1model import classicalV1
import torch.nn as nn
import torch
import numpy as np
from classicalv1.GaborFilters import GaborFilter


class twocells_rot_and_phase_inv(nn.Module):
    def __init__(self):
        super().__init__()
        config = dict(
            inputs_res=[30, 100],
            inputs_fov=[2, 2],
            n_pos_per_deg=1,
            n_ori=6,
            sf=[1],
            n_phase=1,
            sigma_x_values=[0.3],
            sigma_y_values=[0.3],
            with_complex_cells=True,
            with_simple_cells=False,
            noisy_responses=False,
            subset_of_cells_dict={"complex_cells_idx": [0]},
        )
        self.complex_cell = classicalV1(**config)
        self.rotation_inv_cell = RotationInvariantGabor(phase=np.pi / 2)

    def forward(self, x):
        x1 = self.complex_cell(x)
        # print(x1)
        x2 = self.rotation_inv_cell(x)
        # print(x2)
        x = torch.stack([x1.flatten(), x2.flatten()])
        return x


class ComplexCell(classicalV1):
    def __init__(self):
        config = dict(
            inputs_res=[30, 30],
            inputs_fov=[2, 2],
            n_pos_per_deg=1,
            n_ori=6,
            sf=[1],
            n_phase=1,
            sigma_x_values=[0.3],
            sigma_y_values=[0.3],
            with_complex_cells=True,
            with_simple_cells=False,
            noisy_responses=False,
            subset_of_cells_dict={"complex_cells_idx": [2]},
        )
        super().__init__(**config)


class RotationInvariantGabor(nn.Module):
    def __init__(
        self,
        pos=[0, 0],
        sigma_x=0.3,
        sigma_y=0.3,
        sf=1,
        phase=0,
        n_thetas=180,
        res=[30, 30],
        xlim=[-1, 1],
        ylim=[-1, 1],
        max_angle=2 * np.pi,
    ):
        super().__init__()
        thetas = np.linspace(0, max_angle, n_thetas + 1)[:-1]
        filters = torch.stack(
            [
                torch.Tensor(
                    GaborFilter(
                        pos=pos,
                        sigma_x=sigma_x,
                        sigma_y=sigma_y,
                        sf=sf,
                        phase=phase,
                        theta=theta,
                        res=res,
                        xlim=xlim,
                        ylim=ylim,
                    )
                )
                for theta in thetas
            ]
        ).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x):
        x = torch.einsum("bcxy, ncxy-> bn", x, self.filters)
        x, _ = x.max(dim=1)
        return x


class RotationInvariantComplex(nn.Module):
    def __init__(
        self,
        pos=[0, 0],
        sigma_x=0.3,
        sigma_y=0.3,
        sf=1,
        n_thetas=18,
        res=[30, 30],
        xlim=[-1, 1],
        ylim=[-1, 1],
        max_angle=2 * np.pi,
    ):
        super().__init__()
        thetas = np.linspace(0, max_angle, n_thetas + 1)[:-1]
        f1 = torch.stack(
            [
                torch.Tensor(
                    GaborFilter(
                        pos=pos,
                        sigma_x=sigma_x,
                        sigma_y=sigma_y,
                        sf=sf,
                        phase=0,
                        theta=theta,
                        res=res,
                        xlim=xlim,
                        ylim=ylim,
                    )
                )
                for theta in thetas
            ]
        ).unsqueeze(1)
        self.register_buffer("f1", f1)
        f2 = torch.stack(
            [
                torch.Tensor(
                    GaborFilter(
                        pos=pos,
                        sigma_x=sigma_x,
                        sigma_y=sigma_y,
                        sf=sf,
                        phase=np.pi / 2,
                        theta=theta,
                        res=res,
                        xlim=xlim,
                        ylim=ylim,
                    )
                )
                for theta in thetas
            ]
        ).unsqueeze(1)
        self.register_buffer("f2", f2)

    def forward(self, x):
        x1 = torch.einsum("bcxy, ncxy-> bn", x, self.f1)
        x2 = torch.einsum("bcxy, ncxy-> bn", x, self.f2)
        x = torch.sqrt(x1 ** 2 + x2 ** 2)
        x, _ = x.max(dim=1)
        return x


#%%
# from invariant.utils.plot_utils import plot_f

# model = ComplexCell()
# d = {"hparams": {}, "model_state_dict": model.state_dict()}
# torch.save(d, "/project/invariant/presaved_models/ComplexCell/complex_cell30x30.pt")

# model(model.complex_cell_f1s[0].reshape(1,1,30,30))
# # %%
# from invariant.utils.plot_utils import plot_f

# plot_f(model.complex_cell.complex_cell_f1s[0])
# plot_f(model.complex_cell.complex_cell_f2s[0])
# #%%
# plot_f(model.rotation_inv_cell.filters[0])
# plot_f(model.rotation_inv_cell.filters[45])
# plot_f(model.rotation_inv_cell.filters[90])
# plot_f(model.rotation_inv_cell.filters[179])
# # %%
# from classicalv1.GaborFilters import GaborFilter

# N = 30
# gf = torch.stack(
#     [
#         torch.tensor(
#             GaborFilter(
#                 pos=[0, 0],
#                 sigma_x=0.3,
#                 sigma_y=0.3,
#                 sf=1,
#                 theta=t,
#                 phase=phase,
#                 xlim=[-1, 1],
#                 ylim=[-1, 1],
#                 res=[100, 100],
#             )
#         )
#         for t in np.linspace(0, 2 * np.pi, N + 1)[:-1]
#         for phase in np.linspace(0, 2 * np.pi, N + 1)[:-1]
#     ]
# ).reshape(N, N, 100, 100)
# # plot_f(gf)
# # %%
# gf = gf.reshape(-1, 100, 100).reshape(-1, 1, 100, 100).to(torch.float32)
# act = model(gf)
# G = act.reshape(-1, N, N)
# plot_f(G[0])
# plot_f(G[1])
# loss = (
#     -torch.norm(torch.diff(G[0], append=G[0][0, :].reshape(1, -1), dim=0))
#     + torch.norm(torch.diff(G[0], append=G[0][:, 0].reshape(-1, 1), dim=1))
#     + torch.norm(torch.diff(G[1], append=G[1][0, :].reshape(1, -1), dim=0))
#     - torch.norm(torch.diff(G[1], append=G[1][:, 0].reshape(-1, 1), dim=1))
# )
# loss

# %%
