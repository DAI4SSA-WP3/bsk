import numpy as np
from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw

class ImagingDownlinkDynModel(dyn.FullFeaturedDynModel):
    # Define some custom properties to be accessed in the statea
    @property
    def instrument_pointing_error(self) -> float:
        r_BN_P_unit = self.r_BN_P/np.linalg.norm(self.r_BN_P) 
        c_hat_P = self.satellite.fsw.c_hat_P
        return np.arccos(np.dot(-r_BN_P_unit, c_hat_P))
    
    @property
    def solar_pointing_error(self) -> float:
        a = self.world.gravFactory.spiceObject.planetStateOutMsgs[
            self.world.sun_index
        ].read().PositionVector
        a_hat_N = a / np.linalg.norm(a)
        nHat_B = self.satellite.sat_args["nHat_B"]
        NB = np.transpose(self.BN)
        nHat_N = NB @ nHat_B
        return np.arccos(np.dot(nHat_N, a_hat_N))

class ImagingSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speeds_fraction"),
            dict(prop="instrument_pointing_error", norm=np.pi),
            dict(prop="solar_pointing_error", norm=np.pi)
        ),
        obs.Eclipse(),
        obs.OpportunityProperties(
            dict(prop="priority"),
            dict(prop="opportunity_open", norm=5700.0),
            type="target",
            n_ahead_observe=10,
        ),
        obs.OpportunityProperties(
            dict(prop="opportunity_open", norm=5700),
            dict(prop="opportunity_close", norm=5700),
            type="ground_station",
            n_ahead_observe=1,
        ),
        obs.Time(),
    ]
    action_spec = [act.Image(n_ahead_image=10),
        act.Charge(duration=180.0),
        act.Downlink(duration=60.0),
        act.Desat(duration=60.0),]
    dyn_type = ImagingDownlinkDynModel
    fsw_type = fsw.SteeringImagerFSWModel


from bsk_rl.utils.orbital import random_orbit

sat_args = dict(
    # Power
    batteryStorageCapacity=400*3600,
    storedCharge_Init=lambda: 400*3600,#np.random.uniform(300 * 3600 * 0.2, 400 * 3600 * 0.8),
    panelArea=2 * 1.0 * 0.5,
    panelEfficiency=0.20,
    basePowerDraw=-10.0,
    instrumentPowerDraw=-30.0,
    transmitterPowerDraw=-25.0,
    thrusterPowerDraw=-80.0,
    # Data Storage
    dataStorageCapacity=1000 * 8e6,  # MB to bits,
    storageInit=lambda: 0 * 8e6,  #np.random.uniform(0, 5000 * 8e6),
    instrumentBaudRate=0.5e6,
    transmitterBaudRate=-50e6,#-112e6, #baudrate is reduced
    # Attitude
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,
    disturbance_vector=lambda: np.random.normal(scale=0.0001, size=3),
    maxWheelSpeed=6000.0,  # RPM
    wheelSpeeds=lambda: np.random.uniform(-3000, 3000, 3),
    desatAttitude="nadir",
    nHat_B=np.array([0, 0, -1]),  # Solar panel orientation
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
        # ImagingSatellite("EO-2", sat_args),
        # ImagingSatellite("EO-3", sat_args),
        # ImagingSatellite("EO-4", sat_args),
    ]
duration = 5 * 5700.0  # About 2 orbits
env = GeneralSatelliteTasking(
    satellites=multiSat,
    scenario=scene.UniformTargets(1000),
    rewarder=data.UniqueImageReward(),
    time_limit=duration,
    communicator=comm.LOSCommunication(),  # Note that dyn must inherit from LOSCommunication
    log_level="INFO",
    terminate_on_time_limit = True,
    failure_penalty = -1.0,
)
env.reset()

import numpy as np
import torch
from ppoModule import Agent
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pdb

plt.style.use("seaborn-v0_8-darkgrid")
episode_reward = []
plt.ion()
def plot_reward(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_reward, dtype=torch.float)
    if show_result:
        plt.title('Training Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.plot(durations_t.numpy(),color='blue',alpha=0.5)
    # Take M episode averages and plot them too
    M = 20
    if len(durations_t) >= M:
        means = durations_t.unfold(0, M, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(M-1), means))
        plt.plot(means.numpy(), color='blue')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if not show_result:
        plt.draw()
        plt.clf()
    else:
        plt.draw()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01 #was 0.005
LR = 1e-4
EPOCH = 4

# Get the number of state observations
state, info = env.reset()
n_satellites = len(multiSat)

agents = []
for i in range(n_satellites):
    agents.append(Agent(state_dim=env.observation_space[i].shape,
              action_dim=env.action_space[i].n, 
              batch_size=BATCH_SIZE,
              n_epochs=EPOCH,
              policy_clip=0.2,
              gamma=0.99,lamda=0.95, 
              adam_lr=LR))

if torch.cuda.is_available():
    num_episodes = 300
else:
    num_episodes = 20

score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0
best_score = env.reward_range[0]
score_history = []
N = 20

for i_episode in range(num_episodes):
    current_state,info   = env.reset()
    terminated,truncated = False,False
    done  = False
    score = 0
    observation_history = []
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
        observation_history.append(next_state[0])
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            for i in range(n_satellites):
                agents[i].save_models()
        if done:
            state, info = env.reset()
            episode_reward.append(score)
            plot_reward()
            print("Episode complete..")
            break

from datetime import datetime
plot_reward(show_result=True)
save_results_to = '/home/maxsum/Documents/run results basilisk/'
plt.savefig(save_results_to + 'Case_4_Reward'+ datetime.today().strftime('%Y%m%d_%H%M%S') +'.png', dpi = 300)


batt = []
for i in range(len(observation_history)):
    tp=observation_history[i]
    batt.append(tp[1])
plt.figure(2)
plt.plot(batt)
plt.title('Battery Usage within Last Episode')
plt.xlabel('timestep')
plt.ylabel('Battery Level')
plt.savefig(save_results_to + 'Case_4_Battery'+ datetime.today().strftime('%Y%m%d_%H%M%S') +'.png', dpi = 300)

dataStorage = []
for i in range(len(observation_history)):
    tp=observation_history[i]
    dataStorage.append(tp[0])
plt.figure(3)
plt.plot(dataStorage)
plt.title('Data Storage Usage within Last Episode')
plt.xlabel('timestep')
plt.ylabel('Data Storage')
plt.savefig(save_results_to + 'Case_4_DataStorage'+ datetime.today().strftime('%Y%m%d_%H%M%S') +'.png', dpi = 300)