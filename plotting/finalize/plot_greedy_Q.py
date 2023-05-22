import numpy as np
import torch
from matplotlib import pyplot as plt

rc_fonts = {
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    "font.family": "times",
    # "font.family": "Times New Roman",
    "font.size": 10.5,
    'axes.titlesize':8,
    "legend.fontsize":8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    # 'figure.figsize': (2.3, 2.3),
    'figure.figsize': (3, 2.7),
}
plt.rcParams.update(rc_fonts)
# plt.rc('axes', unicode_minus=False)
# plt.tight_layout()
data = torch.load('./inputs_Q1_700.t')
x = torch.arange(-1,1,.005)
gx, gy = torch.meshgrid(x,x)

fig = plt.figure()

ax = plt.axes(projection='3d')

fig.subplots_adjust(left=-0.2, right=1.2, bottom=0, top=1)
d3 = data.cpu().detach().reshape(x.size(0),-1).numpy()
d1, d2 = gx.numpy(), gy.numpy()

# ax.contour3D(d1, d2 ,d3, 50, cmap='binary')

ax.plot_surface(d1, d2, d3,
                cmap='viridis', edgecolor='none')


ax.set_xlabel('rotor1')
ax.set_ylabel('rotor2')
ax.set_yticks([-1,0,1],[-1,0,1])
ax.set_xticks([-1,0,1],[-1,0,1])
ax.set_zlabel('Q value')
ax.view_init(10, 100)
# 调整"rotor1"标签的位置
ax.xaxis.set_label_coords(0, 0)
# for ii in range(0, 360, 5):
#     ax.view_init(elev=10., azim=ii)
#     plt.savefig("./tmp/movie%d.png" % ii)
# plt.show()
# plt.savefig("./2nd.png")

# plt.savefig('./3d_q.pdf', bbox_inches='tight', dpi=300, backend='pdf')
plt.savefig('./3d_q.pdf', bbox_inches='tight', dpi=300, backend='pdf')
fig = plt.figure()
plt.plot(d3[-1,:], linewidth=2)
plt.xticks([0,99,199,299,399],[-1,-0.5,0,0.5,1])
plt.xlabel('rotor1')  # 设置横坐标标签
plt.ylabel('Q value')  # 设置纵坐标标签
plt.savefig('./2d_q.pdf', bbox_inches='tight', dpi=300, backend='pdf')