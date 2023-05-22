from mpl_toolkits import mplot3d
import numpy as np
import torch
from matplotlib import pyplot as plt
# plt.ion()
import torch

from utils.env_utils import env_producer
from model.agent import get_policy_producer,get_q_producer
from utils.default import variant
import utils.pytorch_util as ptu
from utils.pytorch_util import set_gpu_mode

domain = 'swimmer'
seed = 1

set_gpu_mode(use_gpu=True, seed=seed)


expl_env = env_producer(domain, seed)
obs_dim = expl_env.observation_space.low.size
action_dim = expl_env.action_space.low.size

M = variant['layer_size']
q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M, M])
policy_producer = get_policy_producer(
    obs_dim, action_dim, hidden_sizes=[M, M])

policy = policy_producer()
qf1 = q_producer()
qf2 = q_producer()


ob_np = expl_env.reset()
ob = ptu.from_numpy(ob_np)

x = torch.arange(-1,1,.01)
gx, gy = torch.meshgrid(x,x)
action = torch.cat((gx.reshape(-1,1), gy.reshape(-1,1)),1)

ob = ob.to('cuda')
action = action.to('cuda')

# frst
model = 'epoch_0'
path = '/home/chenxing/experiments/model/'+ model+'.ml'
model_state_dict = torch.load(path)
policy.load_state_dict(model_state_dict['policy_state_dict'])
qf1.load_state_dict(model_state_dict['qf1_state_dict'])
qf2.load_state_dict(model_state_dict['qf2_state_dict'])

qf1 = qf1.to('cuda')
data = qf1(ob.reshape(1, -1).expand(action.size(0), len(ob)), action)


fig = plt.figure()
ax = plt.axes(projection='3d')

d1, d2, d3 = gx.numpy(), gy.numpy(), data.cpu().detach().reshape(x.size(0),-1).numpy()

ax.plot_surface(d1, d2, d3,
                cmap='viridis', edgecolor='none')

ax.set_xlabel('rotor1')
ax.set_ylabel('rotor2')
ax.set_zlabel('Q value')
ax.view_init(10, 100)
plt.savefig("./"+model+".png")
# second
model = 'epoch_500'
path = '/home/chenxing/experiments/model/'+ model+'.ml'
model_state_dict = torch.load(path)
policy.load_state_dict(model_state_dict['policy_state_dict'])
qf1.load_state_dict(model_state_dict['qf1_state_dict'])
qf2.load_state_dict(model_state_dict['qf2_state_dict'])

qf1 = qf1.to('cuda')
data = qf1(ob.reshape(1, -1).expand(action.size(0), len(ob)), action)

fig = plt.figure()
ax = plt.axes(projection='3d')
d1, d2, d3 = gx.numpy(), gy.numpy(), data.cpu().detach().reshape(x.size(0),-1).numpy()
ax.plot_surface(d1, d2, d3,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('rotor1')
ax.set_ylabel('rotor2')
ax.set_zlabel('Q value')
ax.view_init(10, 100)
plt.savefig("./"+model+".png")

# 3rd

model = 'epoch_1499'
path = '/home/chenxing/experiments/model/'+ model+'.ml'
model_state_dict = torch.load(path)
policy.load_state_dict(model_state_dict['policy_state_dict'])
qf1.load_state_dict(model_state_dict['qf1_state_dict'])
qf2.load_state_dict(model_state_dict['qf2_state_dict'])

qf1 = qf1.to('cuda')
data = qf1(ob.reshape(1, -1).expand(action.size(0), len(ob)), action)

fig = plt.figure()
ax = plt.axes(projection='3d')
d1, d2, d3 = gx.numpy(), gy.numpy(), data.cpu().detach().reshape(x.size(0),-1).numpy()
ax.plot_surface(d1, d2, d3,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('rotor1')
ax.set_ylabel('rotor2')
ax.set_zlabel('Q value')
ax.view_init(10, 100)
plt.savefig("./"+model+".png")

# 4th

model = 'epoch_1999'
path = '/home/chenxing/experiments/model/'+ model+'.ml'
model_state_dict = torch.load(path)
policy.load_state_dict(model_state_dict['policy_state_dict'])
qf1.load_state_dict(model_state_dict['qf1_state_dict'])
qf2.load_state_dict(model_state_dict['qf2_state_dict'])

qf1 = qf1.to('cuda')
data = qf1(ob.reshape(1, -1).expand(action.size(0), len(ob)), action)

fig = plt.figure()
ax = plt.axes(projection='3d')
d1, d2, d3 = gx.numpy(), gy.numpy(), data.cpu().detach().reshape(x.size(0),-1).numpy()
ax.plot_surface(d1, d2, d3,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('rotor1')
ax.set_ylabel('rotor2')
ax.set_zlabel('Q value')
ax.view_init(10, 100)
plt.savefig("./"+model+".png")