#%%
import torch
import torch.nn as nn
import numpy as np
from classicalv1.GaborFilters import GaborFilter
from invariant.utils.plot_utils import plot_f
from classicalv1.V1cells import complex_cell
from classicalv1.toy_models import *

config = {"phase": 0}
x = RotationInvariantGabor(**config)
path = "/project/invariant/presaved_models/RotationInvariantGabor/RotationInvariantGabor_even_30x30.pt"
d = {"hparams": config, "model_state_dict": x.state_dict()}
torch.save(d, path)

#%%
# from invariant.utils.plot_utils import plot_f
# import numpy as np

# cell = RotationInvariantComplex()
# i = 10
# plot_f(cell.f1[i], cmap="greys", ticks=False)

# plot_f(cell.f2[i], cmap="greys", ticks=False)

# d = {'hparams':{}, 'model_state_dict':cell.state_dict()}
# torch.save(d, '/project/invariant/presaved_models/RotationInvarianComplex/RotationInvarianComplex.pt')
# dg = torch.stack(
#     [
#         torch.Tensor(
#             GaborFilter(
#                 pos=[0, 0],
#                 sigma_x=1000,
#                 sigma_y=1000,
#                 sf=1.3,
#                 phase=phase,
#                 theta=0,
#                 res=[100, 100],
#                 xlim=[-1, 1],
#                 ylim=[-1, 1],
#             )
#         )
#         for phase in np.linspace(0, 2 * np.pi, 100)[:-1]
#     ]
# )
# #%%

# import torchvision.io as tio


# tio.write_video(
#     "drifting_grating_stimulus.mp4", dg.unsqueeze(-1).repeat(1, 1, 1, 3), fps=30
# )


#%%
#%%
# # %%

# import plotly.express as px
# import plotly.io as pio

# pio.renderers.default = "notebook"
# #%%
# idx = 0
# img = x.detach().numpy()
# fig = px.imshow(
#     img[idx].squeeze(),
#     animation_frame=0,
#     binary_string=True,
#     labels=dict(animation_frame="slice"),
# )
# fig.show()
# # %%
# idx = 10
# img = x.detach().numpy()
# fig = px.imshow(
#     img[:, idx].squeeze(),
#     animation_frame=0,
#     binary_string=True,
#     labels=dict(animation_frame="slice"),
# )
# fig.show()


# # #%%

# # import matplotlib
# # # Usually we use `%matplotlib inline`. However we need `notebook` for the anim to render in the notebook.
# # %matplotlib notebook

# # import random
# # import numpy as np

# # import matplotlib
# # import matplotlib.pyplot as plt

# # import matplotlib.animation as animation


# # fps = 30

# # # First set up the figure, the axis, and the plot element we want to animate
# # fig = plt.figure( figsize=(8,8) )
# # dgn = dg.numpy()
# # a = dgn[0]
# # im = plt.imshow(a, interpolation='none', aspect='auto', vmin=-np.max(np.abs(dgn)), vmax=np.max(np.abs(dgn)), cmap='Greys_r')
# # plt.xticks([])
# # plt.yticks([])
# # def animate_func(i):
# #     if i % fps == 0:
# #         print( '.', end ='' )

# #     im.set_array(dgn[i])
# #     return [im]

# # anim = animation.FuncAnimation(fig,
# #                                animate_func,
# #                                frames = len(dg),
# #                                interval = 1000 / fps, # in ms
# #                                )

# # anim.save('drifting_grating.gif', fps=fps, writer="pillow")

# # print('Done!')

# # #%%
# # x =GaborFilter([0,0], sigma_x= 100,sigma_y=100,sf=1.3,phase=0, theta=0, res=[101,101], xlim=[-1, 1], ylim=[-1, 1])
# # plot_f(x)
# # #%%
# # img = np.stack([x.detach().numpy()[i, i] for i in range(len(x))])
# # fig = px.imshow(
# #     img.squeeze(),
# #     animation_frame=0,
# #     binary_string=True,
# #     labels=dict(animation_frame="slice"),
# # )
# # fig.show()


# # # %%
# # img1 = torch.stack([x[i, i] for i in range(len(x))])

# # from invariant.utils.cosine_similarity import cosine_similarity
# # from invariant.utils.plot_utils import plot_f

# # plot_f(cosine_similarity(img1.flatten(1)), title="img[i, i]")


# # #%% Plot i+j, i
# # for j in range(20):
# #     plot_f(
# #         cosine_similarity(
# #             torch.stack([x[(i + j) % len(x), i] for i in range(len(x))]).flatten(1),
# #         ),
# #         title=f"img[i+{j}, i]",
# #     )
# # # %% i, i,+j
# # for j in range(20):
# #     plot_f(
# #         cosine_similarity(
# #             torch.stack([x[i, (i + j) % len(x)] for i in range(len(x))]).flatten(1)
# #         ),
# #         title=f"img[i, i+{j}]",
# #     )

# # # %%


# # x = torch.zeros(100, 100)
# # x[:, 50:] = 1
# # plot_f(x, ticks=False, cmap="greys")

# # x = torch.zeros(100, 100)
# # x[:, 50:] = -1
# # plot_f(x, ticks=False, cmap="greys")
# # # %%


# # # %%

# # # %%

# %%
