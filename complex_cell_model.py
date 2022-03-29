#%%
import torch
from torchV1model import classicalV1
from invariant.utils.plot_utils import plot_f
from GaborFilters import GaborFilter
from V1cells import complex_cell
from invariant.utils.img_transf import Normalize
import numpy as np
import matplotlib.pyplot as plt


def get_MEIs_for_complex_cell(cs, phases, std=0.1, mean=0):
    normalize = Normalize(mean=mean, std=std, dim=[1, 2, 3])
    gfs = torch.stack(
        [
            torch.Tensor(
                GaborFilter(
                    pos=cs.pos,
                    sf=cs.sf,
                    phase=phase,
                    theta=cs.theta,
                    sigma_x=cs.sigma_x,
                    sigma_y=cs.sigma_y,
                    res=cs.res,
                    xlim=cs.xlim,
                    ylim=cs.ylim,
                )
            ).unsqueeze(0)
            for phase in phases
        ]
    )
    gfs = normalize(gfs)
    return gfs


# %%
single_complex_cell_V1_model = classicalV1(
    inputs_res=[100, 100],
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

print(
    f"In the model are present {len(single_complex_cell_V1_model.model.complex_cells)} complex cells and "
    + f"{len(single_complex_cell_V1_model.model.simple_cells)} simple cells"
)
plot_f(
    single_complex_cell_V1_model.model.complex_cells[0].f1, title="complex_cell_filter"
)
cs = single_complex_cell_V1_model.model.complex_cells[0]

phases = np.linspace(0, 2 * np.pi, 20)
gfs = get_MEIs_for_complex_cell(cs, phases=phases, std=0.01)

activations = []
for gf in gfs:
    activation = cs.get_response(gf.squeeze().data.numpy())
    plot_f(gf, title=activation)
    activations.append(activation)
#%%
plt.plot(phases, activations, label="activation over phase")
plt.show()
plt.plot(phases, activations, label="activation over phase")
plt.ylim(0)
plt.show()


def cosine_similarity(tensor):
    tensor_normed = tensor / torch.norm(tensor, p=2, dim=0, keepdim=True)
    return tensor_normed.T @ tensor_normed


# %%
cosine_similarity(gfs.flatten(1))
plot_f(-cosine_similarity(gfs.flatten(1).T), title="cosine dissimilarity")
p2dist_matrix_images = torch.cdist(gfs.flatten(1), gfs.flatten(1), p=2)
plot_f(p2dist_matrix_images, vmin=0, title="p=2")
p1dist_matrix_images = torch.cdist(gfs.flatten(1), gfs.flatten(1), p=1)
plot_f(p1dist_matrix_images, vmin=0, title="p=1")

#%%
phases = torch.tensor(phases)
pdist_matrix_phase = torch.cdist(phases.unsqueeze(1), phases.unsqueeze(1), p=2)
plot_f(pdist_matrix_phase)

# #%%
# pdist_matrix = pdist_matrix_images * pdist_matrix_phase
# scale_the_average = (batch_size ** 2 - batch_size) / (batch_size ** 2)

# pdist_matrix.mean() * scale_the_average
# # %%
# %%
gfs2 = gfs[: int(len(gfs) / 2)]
plot_f(-cosine_similarity(gfs2.flatten(1).T), title="cosine dissimilarity")
p2dist_matrix_images = torch.cdist(gfs2.flatten(1), gfs2.flatten(1), p=2)
plot_f(p2dist_matrix_images, title="p=2")
p1dist_matrix_images = torch.cdist(gfs2.flatten(1), gfs2.flatten(1), p=1)
plot_f(p1dist_matrix_images, title="p=1")

#%%
phases = torch.tensor(phases)[: int(len(gfs) / 2)]
pdist_matrix_phase = torch.cdist(phases.unsqueeze(1), phases.unsqueeze(1), p=2)
plot_f(pdist_matrix_phase, title="phases")

# %%
