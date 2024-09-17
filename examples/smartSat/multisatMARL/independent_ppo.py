## EO Satellite Environment Definition:
from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw

class ImagingSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.OpportunityProperties(
            dict(prop="priority"),
            dict(prop="opportunity_open", norm=5700.0),
            n_ahead_observe=10,
        )
    ]
    action_spec = [act.Image(n_ahead_image=10)]
    dyn_type = dyn.FullFeaturedDynModel
    fsw_type = fsw.SteeringImagerFSWModel


from bsk_rl.utils.orbital import random_orbit

sat_args = dict(
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,
    batteryStorageCapacity=1e9,
    storedCharge_Init=1e9,
    dataStorageCapacity=1e12,
    u_max=0.4,
    K1=0.25,
    K3=3.0,
    omega_max=0.087,
    servo_Ki=5.0,
    servo_P=150 / 5,
    oe=lambda: random_orbit(alt=500),
)


from bsk_rl import GeneralSatelliteTasking
multiSat=[
        ImagingSatellite("EO-1", sat_args),
        ImagingSatellite("EO-2", sat_args),
        ImagingSatellite("EO-3", sat_args),
        ImagingSatellite("EO-4", sat_args),
    ]
duration = 5 * 5700.0  # About 5 orbits

env = GeneralSatelliteTasking(
    satellites = multiSat,
    scenario   = scene.UniformTargets(1000),
    time_limit = duration,
    failure_penalty = -1.0,
    rewarder     = data.UniqueImageReward(),
    communicator = comm.LOSCommunication(),  # Note that dyn must inherit from LOSCommunication
    log_level    = "INFO",
    terminate_on_time_limit = True,
)
env.reset()

#######################################################
import numpy as np
import torch
from ppoModule import Agent
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

## Plot function:
plt.style.use("seaborn-v0_8-darkgrid")
episode_reward = []
plt.ion()
def plot_reward(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_reward, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take M episode averages and plot them too
    M = 10
    if len(durations_t) >= M:
        means = durations_t.unfold(0, M, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(M-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if not show_result:
        plt.draw()
        plt.clf()
    else:
        plt.draw()

######################################################################
## Learing Parameters
BATCH_SIZE = 256
GAMMA      = 0.99
EPS_START  = 0.9
EPS_END    = 0.05
EPS_DECAY  = 1000
TAU   = 0.01   # was 0.005
LR    = 1e-4
EPOCH = 4
N     = 20     # update periode

# Get the number of state observations
state, info = env.reset()
n_satellites = len(multiSat)

# Agents definition:
agents = []
for i in range(n_satellites):
    agents.append(Agent(state_dim=env.observation_space[i].shape,
              action_dim=env.action_space[i].n, 
              batch_size=BATCH_SIZE,
              n_epochs=EPOCH,
              policy_clip=0.2,
              gamma=0.99,lamda=0.95, 
              adam_lr=LR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    num_episodes = 20
else:
    num_episodes = 20

score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0
best_score = env.reward_range[0]
score_history = []

## Episodes loop
for i_episode in range(num_episodes):
    current_state,info   = env.reset()
    terminated,truncated = False,False
    done  = False
    score = 0
    
    while not done:
        actions = []
        probs   = []
        vals    = []
        for i in range(n_satellites):
            act, prob, val = agents[i].choose_action(current_state[i])
            actions.append(act)
            probs.append(prob)
            vals.append(val)
        action = tuple(actions)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = 1 if (terminated or truncated) else 0
        n_steps += 1
        score += reward
        for i in range(n_satellites):
            agents[i].store_data(current_state[i], actions[i], probs[i], vals[i], reward, done)

        if n_steps % N == 0:
            for i in range(n_satellites):
                agents[i].learn()
            learn_iters += 1
           
        current_state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            for i in range(n_satellites):
                agents[i].save_models()
        if done:
            # state, info = env.reset()
            episode_reward.append(score)
            plot_reward()
            print("Episode:" + str(i_episode) + " complete..")
            break
        
plot_reward(show_result=True)
save_results_to = '/home/maxsum/Documents/run results basilisk/'
plt.savefig(save_results_to + 'multiSat_PPO_200.png', dpi = 300)
plt.ioff()
plt.show()