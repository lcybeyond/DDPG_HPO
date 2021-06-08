import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import mean_squared_error
import numpy as np
import utils
from model_xgb import construct_xgb
from read_data import read_data


class Actor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, s):
        x = torch.relu(self.linear1(s))
        x = torch.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class env(torch.nn.Module):
    def __init__(self):
        # 初始化父类
        torch.nn.Module.__init__(self)
        self.encoder = torch.nn.Linear(100, 100)
        self.lstm = torch.nn.LSTMCell(100, 100)
        self.decoder = torch.nn.Linear(100, 10)
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(torch.zeros(key, 100), requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)
        self.X_train = read_data()[0]
        self.X_test = read_data()[1]
        self.y_train = read_data()[2]
        self.y_test = read_data()[3]

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)

    def reset(self):
        inputs = self.static_inputs[1]
        self.hidden = self.static_init_hidden[1]
        return (inputs, self.hidden)

    def forward(self, inputs, hidden):
        embed = self.encoder(inputs)
        hx, cx = self.lstm(embed, hidden)
        logits = self.decoder(hx)
        logits = (2.5 * torch.sigmoid(logits))
        return logits, (hx, cx)

    def step(self, action):
        xgb = construct_xgb(action)
        xgb.fit(self.X_train.astype('float') / 256, self.y_train)
        pred = xgb.predict(self.X_test)
        Reward = -mean_squared_error(self.y_test, pred)

        inputs = self.static_inputs[1]
        logits, hidden = self.forward(inputs, self.hidden)
        self.hidden = hidden
        state = (inputs, hidden)

        return state, Reward, False

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, 100)
        return (utils.get_variable(zeros, requires_grad=False),
                utils.get_variable(zeros.clone(), requires_grad=False))


class Skylark_DDPG():
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = 10000
        self.batch_size = 32

        s_dim = 300
        a_dim = 10

        self.actor = Actor(s_dim, 256, a_dim)
        self.actor_target = Actor(s_dim, 256, a_dim)
        self.critic = Critic(s_dim+a_dim, 256, 1)
        self.critic_target = Critic(s_dim+a_dim, 256, 1)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        self.REWARD=[]

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.X_train = read_data()[0]
        self.X_test = read_data()[1]
        self.y_train = read_data()[2]
        self.y_test = read_data()[3]

    def act(self, s0):
        input,hidden=s0
        hx,cx=hidden
        s0=torch.cat([input,hx,cx],1)
        a0 = self.actor(s0).squeeze(0).detach().numpy()

        return a0

    def put(self, *transition):
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        s0, a0, r1, s1=transition

        input, hidden = s0
        hx, cx = hidden
        s0 = torch.cat([input, hx, cx], 1)
        s0=s0.detach().numpy()

        input, hidden = s1
        hx, cx = hidden
        s1 = torch.cat([input, hx, cx], 1)
        s1 = s1.detach().numpy()

        self.buffer.append((s0,a0,r1,s1))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float).squeeze(1)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float).squeeze(1)

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            action=self.actor(s0)
            xgb = construct_xgb(action.detach().numpy()[0])
            xgb.fit(self.X_train.astype('float') / 256, self.y_train)
            pred = xgb.predict(self.X_test)
            reward = mean_squared_error(self.y_test, pred)
            print(reward)
            self.REWARD.append(reward)


            loss = -torch.mean( self.critic(s0, self.actor(s0)) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

    def train(self, num_episodes):
        for i in range(1, num_episodes):
            s0 = self.env.reset()
            episode_reward = 0

            for t in range(1, 1000):
                # self.env.render()
                a0 = self.act(s0)
                s1, r1, done = self.env.step(a0)
                self.put(s0, a0, r1, s1)

                episode_reward += r1
                s0 = s1

                self.learn()

            print('Episode {} : {}'.format(i, episode_reward))
        print(len(self.REWARD))
        plt.plot(range(968), self.REWARD)
        plt.show()
        np.savetxt("./result.txt", self.REWARD)






environment=env()
DDPG=Skylark_DDPG(environment)
DDPG.train(2)