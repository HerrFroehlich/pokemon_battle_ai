from src.training.competitions import Competition_1vs1, CompetitionConfig, AgentConfig
from src.models.dqn_network import DQNConfig
from src.models.rewards import *

import matplotlib.pyplot as plt
import numpy as np

import torch

# -------------- Setup NN
nnconf = DQNConfig()
#nnconf.D_IN = 1024 ->  setup by agent
#nnconf.D_OUT = 1024 ->  setup by agent
nnconf.HIDDENLAYER_SIZES = [512,256,64,256]
nnconf.HIDDENLAYER_NETWORK_TYPE = torch.nn.Linear # layer type
nnconf.ACTIVATION_FUNCTION=torch.nn.functional.relu # activator function of each layer
# -------------- Setup Agents
aconf = AgentConfig(NETWORK_CONFIG=nnconf)
aconf.MEMORY_SIZE = 10000 # NOF stored states in memory
aconf.BATCH_SIZE = 128 # NOF batches used for optimizing
aconf.EPS_START = 0.9 # epsilon start for expontential random decay function
aconf.EPS_END = 0.05# epsilon end for expontential random decay function
aconf.EPS_DECAY = 200 # decay gradient for expontential random decay function
aconf.GAMMA = 0.999
aconf.TARGET_UPDATE = 50 # after how many turns update the target NN
aconf.REWARD_FNCT = reward_hp_diff
aconf.LOSS_FNCT = torch.nn.functional.smooth_l1_loss
aconf.OPTIMIZER = torch.optim.RMSprop

# -------------- Setup Competition
conf = CompetitionConfig(AGENTCONFIG=aconf)
conf.N_BATTLES = 10000
conf.N_BATTLES_WITH_SAME_TEAM = 50
conf.ENABLE_STATS = True
conf.STATS_LOG_STEP = 50
conf.DEBUG = False
print("#- Using config:")
print(conf)


c = Competition_1vs1(conf)
c.run()
stats = c.get_stats()

fig, ax = plt.subplots(1,2)
fig2, ax2 = plt.subplots(1,2)

for i in range(2):
    ax[i].set_xlabel("Nof Battles")
    # ax[i].set_ylabel("m²")
    ax[i].set_title("TEAM%d" % (i+1))
    line, = ax[i].plot(stats.timestamps,stats.avg_reward[:,i])
    line.set_label("Avg Reward")
    line, = ax[i].plot(stats.timestamps,stats.avg_loss[:,i])
    line.set_label("Avg loss")
    ax[i].legend()

    ax2[i].set_xlabel("Nof Battles")
    # ax2[i].set_ylabel("m²")
    ax2[i].set_title("TEAM%d" % (i+1))
    line, = ax2[i].plot(stats.timestamps, np.cumsum(stats.effective_cnt[:,i]))
    line.set_label("Cum. effective")
    line, = ax2[i].plot(stats.timestamps, np.cumsum(stats.ineffective_cnt[:,i]))
    line.set_label("Cum. ineffective")
    line, = ax2[i].plot(stats.timestamps, np.cumsum(stats.normal_cnt[:,i]))
    line.set_label("Cum. normal")
    line, = ax2[i].plot(stats.timestamps,np.cumsum(stats.random_cnt[:,i]))
    line.set_label("Cum.  Nof random choices")
    ax2[i].legend()



plt.show()