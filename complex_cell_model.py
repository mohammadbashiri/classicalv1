#%%
import torch
from torchV1model import classicalV1
from invariant.utils.plot_utils import plot_f
from GaborFilters import GaborFilter
from V1cells import complex_cell
from invariant.utils.intermediate_transf import Normalize
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


# %% TWO COMPLEX CELLS MODEL
config = dict(
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
    subset_of_cells_dict={"complex_cells_idx": [0, 3]},
)
single_complex_cell_V1_model = classicalV1(**config)

print(
    f"In the model are present {len(single_complex_cell_V1_model.model.complex_cells)} complex cells and "
    + f"{len(single_complex_cell_V1_model.model.simple_cells)} simple cells"
)
plot_f(
    single_complex_cell_V1_model.model.complex_cells[0].f1, title="complex_cell_filter"
)
plot_f(
    single_complex_cell_V1_model.model.complex_cells[1].f1, title="complex_cell_filter"
)

#%% store model
d = {"hparams": config, "model_state_dict": single_complex_cell_V1_model.state_dict()}
torch.save(
    d,
    "/project/invariant/presaved_models/classicalV1/two_complex_cell_hparams_and_state_dict.pt",
)
#%%

cs = single_complex_cell_V1_model.model.complex_cells[0]
size = 201
phases = np.linspace(0, 2 * np.pi, size + 1)
gfs = get_MEIs_for_complex_cell(cs, phases=phases, std=0.01)

activations = []
for gf in gfs:
    activation = cs.get_response(gf.squeeze().data.numpy())
    #     plot_f(gf, title=activation)
    activations.append(activation)


def ring_dist(imgs):
    size = len(imgs)
    pd = nn.PairwiseDistance()
    one_step_distance = pd(
        imgs.flatten(1), torch.roll(imgs.flatten(1), shifts=-1, dims=0)
    )
    rolled_one_step_distance = torch.stack(
        [torch.roll(one_step_distance, shifts=i, dims=0) for i in range(size)]
    )
    opposite_dir_distances = torch.flip(
        rolled_one_step_distance[:, int(size / 2) + 1 :], dims=[1]
    )
    rolled_distances = rolled_one_step_distance[:, : int(size / 2)]

    cumdist = torch.cumsum(rolled_distances, dim=1)
    opposite_dir_cumdist = torch.flip(
        torch.cumsum(opposite_dir_distances, dim=1), dims=[1]
    )
    zerodist = torch.zeros(size, 1).to("cuda")
    ring_distances = torch.cat([zerodist, cumdist, opposite_dir_cumdist], dim=1)
    for i in range(1, size):
        ring_distances[i] = torch.roll(ring_distances[i], shifts=i, dims=0)
    return ring_distances


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
plot_f(cosine_similarity(gfs.flatten(1).T), title="cosine similarity")
p2dist_matrix_images = torch.cdist(gfs.flatten(1), gfs.flatten(1), p=2)
plot_f(p2dist_matrix_images, vmin=0, title="p=2")
p1dist_matrix_images = torch.cdist(gfs.flatten(1), gfs.flatten(1), p=1)
plot_f(p1dist_matrix_images, vmin=0, title="p=1")

#%%
phases = torch.tensor(phases)
pdist_matrix_phase = torch.cdist(phases.unsqueeze(1), phases.unsqueeze(1), p=2)
plot_f(pdist_matrix_phase)
plot_f(torch.cos(pdist_matrix_phase))
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
phases = torch.linspace(0, np.pi * 2, 1000)
pdist_matrix_phase = torch.cdist(phases.unsqueeze(1), phases.unsqueeze(1), p=2)
plot_f(pdist_matrix_phase, title="pairwise distances of phases", ticks=False)
plot_f(
    torch.cos(pdist_matrix_phase),
    title="cosine of pairwise distances of phases",
    ticks=False,
)
# %%

# %%

# %%
target = torch.cos(pdist_matrix_phase)
gt = cosine_similarity(gfs.flatten(1).T)
# %%
plot_f(target)
plot_f(gt)
plot_f((target - gt) ** 2)
# %% RING distances
import torch.nn as nn

size = 2000
phases = np.linspace(0, 2 * np.pi, size + 1)[:-1]
gfs = get_MEIs_for_complex_cell(cs, phases=phases)


d = ring_distances(gfs)
import matplotlib.pyplot as plt

# plt.plot(phases, d, "*-")
# plt.title("one step distance between MEI gabor filters for complex cell")
# plt.show()
plot_f(d)

phases_prep = torch.tensor(phases).unsqueeze(1)
target = torch.cdist(phases_prep, phases_prep, p=1)
target[target > np.pi] = np.pi - (target[target > np.pi] % np.pi)
plot_f(target)
#%%
target_n = target / torch.sum(target)
d_n = d / torch.sum(d)
plot_f(d_n)
plot_f(target_n)
plot_f(d_n - target_n)


#%%
# get target with phases


#%%

#%%
images = torch.stack([torch.cos(x), torch.sin(x)]).swapaxes(0, 1)


#%%
pairwise_distance = nn.PairwiseDistance()

one_step_distance = pairwise_distance(
    images.flatten(1), torch.roll(images.flatten(1), shifts=-1, dims=0)
)
rolled_one_step_distance = torch.stack(
    [torch.roll(one_step_distance, shifts=i, dims=0) for i in range(size)]
)
print(rolled_one_step_distance)
#%%
opposite_dir_distances = torch.flip(
    rolled_one_step_distance[:, int(size / 2) + 1 :], dims=[1]
)
rolled_distances = rolled_one_step_distance[:, : int(size / 2)]

cumdist = torch.cumsum(rolled_distances, dim=1)
opposite_dir_cumdist = torch.flip(torch.cumsum(opposite_dir_distances, dim=1), dims=[1])
zerodist = torch.zeros(size, 1)
ring_distances = torch.cat([zerodist, cumdist, opposite_dir_cumdist], dim=1)
for i in range(1, size):
    ring_distances[i] = torch.roll(ring_distances[i], shifts=i, dims=0)

print(ring_distances)

#%%

# %%
