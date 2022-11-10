import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # print("input : "+str(x))
        x = F.relu(self.fc1(x))
        # print("hidden layer is :"+str(x))
        t = self.fc2(x)
        # print("output layer is : "+str(t))
        return F.softmax(t)


class Reinforce:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        # 4 , 128 , 2 , 0.001 , 0.98 , cpu
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor(state, dtype=torch.float)
        probs = self.policy_net(state)
        # print("probs are :"+ str(probs))
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        # print(probs,action_dist.__str__(),action,sep=" |||| ")
        return action.item()

    # 对于每一条路径，计算
    def update(self, transition_dict):

        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        print(reward_list,state_list,action_list,sep="\n")
        # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # [state1,state2,state3 ...... ]
        # action : [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0]

        G = 0
        self.optimizer.zero_grad() # 导数变成0

        # range : 0 - 20 ， 长度为21.
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]

            state = torch.tensor([state_list[i]], dtype=torch.float)   # take each state
            # tensor([[ 0.0690,  1.3632, -0.1997, -2.2066]])
            action = torch.tensor([action_list[i]]).view(-1, 1)# -1表示不确定要分成几行,反正只分成一列.
            # tensor([[0]]) # take each action
            # output is : [[0.4212,0.5788]]
            # print("different between gather : "+str(self.policy_net(state))+" "+str(self.policy_net(state).gather(1, action))
            #       +" my thought : " + str(self.policy_net(state)[0][action[0][0]])
            #       )
            # log pi(s,a)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = - log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        # loss is tensor([[11.3805]], grad_fn=<MulBackward0>)
        # print(loss.__class__,loss.__str__())

        self.optimizer.step()  # 梯度下降


learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v1"
env = gym.make(env_name,render_mode="human")
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = Reinforce(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

return_list = []
is_first_state = False
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()[0]
            done = False
            while not done:
                # state : (array([-0.03032472,  0.0038686 ,  0.04191616,  0.00304976], dtype=float32), {})
                action = agent.take_action(state)
                next_state, reward, done, _ ,useless = env.step(action)
                # print("new step :"," step info : "+str(next_state),reward,done)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
