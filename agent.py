import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory # USED during Training
import itertools
import yaml 
import torch
import torch.nn as nn
import torch.optim as optim
import random

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Agent:
    #constructor => store parameter set
    def __init__(self,param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f: #read mode 'f' from parameter yaml file
            all_param_set = yaml.safe_load(f) # safeloaded file
            params = all_param_set[param_set] #store our params from all param set

        #Agent gets access of all params
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"] 
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.reward_threshold = params["reward_threshold"]

        self.network_sync_rate = params["network_sync_rate"]
        self.mini_batch_size = params["mini_batch_size"]
                

        self.loss_fn = nn.MSELoss() #loss func
        self.optimizer = None

    #create env in run func
    def run(self, is_training = True, render = False): 

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.oberservation_space.shape[0] #input dim
        num_actions = env.action_space.n # output dim


         #make deep q policy network
        policy_dqn = DQN(num_states, num_actions).to(device)

        
        #size defined for memory
        if is_training:
            memory = ReplayMemory(self.replay_memory_size) #dynamic
            epsilon = self.epsilon_init # initialise epsilon with 1 , decayed slowly

        #training start for all episodes
        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_rewards = 0
            terminated = False

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() #explore
                    action = torch.tensor(action, dtype=torch.long, device=device)   
                else:
                    with torch.no_grad(): #no computation of gradients
                         action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() #exploit (returns best q val for most optimal action) , no learning
                                                                        
                next_state, reward, terminated, _, _ = env.step(action.item()) #.item() => extracting action val from tensors

                #convert to tensor 
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                        memory.append(state, action, new_state, reward, terminated) #new exp appended in memory


                state = new_state
                episode_rewards += rewards
            
            print(f"episode : {episode+1} with total rewards : {episode_rewards} & epsilon : {epsilon}")




            #EPSILON DECAY - after every episode
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min) #the value after epsilon decay or min threshold val for epsilon -> highest will be th next epsilon val of our next episode/iteration

            # env.close() => commented cuz we want to manually stop