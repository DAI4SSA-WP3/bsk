import numpy as np
from bsk_rl import act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw

import numpy as np
import torch
from ppoModule import Agent
import matplotlib.pyplot as plt

class ScanningDownlinkDynModel(dyn.ContinuousImagingDynModel, dyn.GroundStationDynModel):
    # Define some custom properties to be accessed in the state
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

class ScanningSatellite(sats.AccessSatellite):
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
            dict(prop="opportunity_open", norm=5700),
            dict(prop="opportunity_close", norm=5700),
            type="ground_station",
            n_ahead_observe=1,
        ),
        obs.Time(),
    ]
    action_spec = [
        act.Scan(duration=180.0),
        # act.Charge(duration=180.0), # deactivated for Case-1
        act.Downlink(duration=60.0),
        act.Desat(duration=60.0),
    ]
    dyn_type = ScanningDownlinkDynModel
    fsw_type = fsw.ContinuousImagingFSWModel

sat = ScanningSatellite(
    "Scanner-1",
    sat_args=dict(
        # Data
        dataStorageCapacity=5000 * 8e6,  # MB to bits
        storageInit=lambda: 0, #np.random.uniform(0, 5000 * 8e6),
        instrumentBaudRate=0.5e6,
        transmitterBaudRate=-112e6,
        # Power
        batteryStorageCapacity=400 * 3600,  # Wh to W*s
        storedCharge_Init=lambda: 400 * 3600,  #np.random.uniform(400 * 3600 * 0.2, 400 * 3600 * 0.8),
        panelArea=0.0, #2 * 1.0 * 0.5,
        panelEfficiency=0.0, #0.20,
        basePowerDraw=-10.0,
        instrumentPowerDraw=-30.0,
        transmitterPowerDraw=-25.0,
        thrusterPowerDraw=-80.0,
        # Attitude
        imageAttErrorRequirement=0.1,
        imageRateErrorRequirement=0.1,
        disturbance_vector=lambda: np.random.normal(scale=0.0001, size=3),
        maxWheelSpeed=6000.0,  # RPM
        wheelSpeeds=lambda: np.random.uniform(-3000, 3000, 3),
        desatAttitude="nadir",
        nHat_B=np.array([0, 0, -1]),  # Solar panel orientation
    )
)

duration = 2 * 5700.0  # About 2 orbits
env_args = dict(
    satellite=sat,
    scenario=scene.UniformNadirScanning(value_per_second=1/duration),
    rewarder=data.ScanningTimeReward(),
    time_limit=duration,
    failure_penalty=-1.0,
    terminate_on_time_limit=True,
)

from bsk_rl import SatelliteTasking
env = SatelliteTasking(**env_args, log_level="INFO")


# env.reset()
# env.reset(seed=RANDOM_SEED)
# print(env.observation_space)
# print(env.action_space.n)



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
    plt.ylabel('Rewards')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    M = 20
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
RANDOM_SEED = 11
BATCH_SIZE = 5
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01 #was 0.005
LR = 0.00003
EPOCH = 4
ENTROPY_COEFF = 0.05

# Get the number of state observations
state, info = env.reset()
# state, info = env.reset(seed=RANDOM_SEED) # --> Activate this reset to manually seed the environment

ppoAgent = Agent(state_dim=env.observation_space.shape,
              action_dim=env.action_space.n,
            #   entropy_coeff=ENTROPY_COEFF, 
              batch_size=BATCH_SIZE,
              n_epochs=EPOCH,
              policy_clip=0.1,
              gamma=GAMMA,lamda=0.95, 
              adam_lr=LR)

if torch.cuda.is_available():
    num_episodes = 100
else:
    num_episodes = 20

learn_iters = 0
avg_score = 0
n_steps = 0
best_score = env.reward_range[0]
score_history = []
reward_history = []
observation_history = []
N = 10

for i_episode in range(num_episodes):
    current_state,info   = env.reset()
    # current_state,info   = env.reset(seed=RANDOM_SEED) # --> Activate this reset to manually seed the environment
    terminated,truncated = False,False
    done  = False
    score = 0
    reward_history = []
    observation_history = []
    while not done:
        act, prob, val = ppoAgent.choose_action(current_state)
        next_state, reward, terminated, truncated, info = env.step(act)
        done = 1 if (terminated or truncated) else 0
        n_steps += 1
        score += reward

        ppoAgent.store_data(current_state, act, prob, val, reward, done)

        if n_steps % N == 0:
            ppoAgent.learn()
            learn_iters += 1
        current_state = next_state

        observation_history.append(next_state)
        reward_history.append(reward)
    score_history.append(score)
    avg_score = np.mean(score_history[-10:])
        
    if avg_score > best_score:
        best_score = avg_score
        ppoAgent.save_models()

    plt.figure(2)
    plt.plot(reward_history)
    plt.show()
    episode_reward.append(score)
    print('episode', i_episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
        'time_steps', n_steps, 'learning_steps', learn_iters)


from datetime import datetime
# save_results_to = '/home/maxsum/Documents/run results basilisk/'

batt = []
for i in range(len(observation_history)):
    tp=observation_history[i]
    batt.append(tp[1])

plt.figure(3)
plt.title("Battery Consumption")
plt.plot(batt)
# plt.savefig(save_results_to + 'Case_1_Battery'+ datetime.today().strftime('%Y%m%d_%H%M%S') +'.png', dpi = 300)
plt.show()

storage = []
for i in range(len(observation_history)):
    tp=observation_history[i]
    storage.append(tp[1])
plt.figure(4)
plt.title("Data Storage Usage")
plt.plot(storage)
# plt.savefig(save_results_to + 'Case_1_Battery'+ datetime.today().strftime('%Y%m%d_%H%M%S') +'.png', dpi = 300)
plt.show()


plot_reward(show_result=True)
# plt.savefig(save_results_to + 'Case_1_Reward'+ datetime.today().strftime('%Y%m%d_%H%M%S') +'.png', dpi = 300)