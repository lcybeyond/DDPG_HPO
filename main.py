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
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear4 = torch.nn.Linear(hidden_size, output_size)
        pass
    def forward(self, s):
        x = torch.relu(self.linear1(s))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear4 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        a= torch.reshape(a,(32,1))
        x = torch.cat([s, a], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)

        return x


class env(torch.nn.Module):
    def __init__(self):
        # 初始化父类
        torch.nn.Module.__init__(self)
        self.encoder = torch.nn.Linear(100, 100)
        self.lstm = torch.nn.LSTMCell(100, 100)
        self.decoder = torch.nn.Linear(100, 1)
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(np.random.normal(loc=0.0, scale=5.0, size=(1,100))  , requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)
        self.X_train = read_data()[0]
        self.X_test = read_data()[1]
        self.y_train = read_data()[2]
        self.y_test = read_data()[3]

        self.baseline=None

        self.action_buffer=[]
    def reset_parameters(self):
        for param in self.parameters():
            param.data.normal_(0, 5.0)
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

        self.action_buffer.append(action)
        if len(self.action_buffer)<10:
            Reward=-1
            done=False
        else:
            xgb = construct_xgb(self.action_buffer)
            xgb.fit(self.X_train.astype('float') / 256, self.y_train)
            pred = xgb.predict(self.X_test)
            Reward = 9-mean_squared_error(self.y_test, pred)
            self.update_i=0
            self.action_buffer=[]
            done=True
        # if self.baseline is None:
        #     self.baseline = Reward
        # else:
        #     self.baseline = 0.95 * self.baseline + (1 - 0.95) * Reward
        #
        # Reward=Reward-self.baseline

        inputs = self.static_inputs[1]
        logits, hidden = self.forward(inputs, self.hidden)
        self.hidden = hidden
        state = (inputs, hidden)

        return state, Reward, done

    def init_hidden(self, batch_size):
        zeros = np.random.normal(loc=0.0, scale=5.0, size=(1,100))
        return (utils.get_variable(zeros, requires_grad=False),
                utils.get_variable(zeros, requires_grad=False))


class Skylark_DDPG():
    def __init__(self):
        self.gamma = 0.99
        self.actor_lr = 0.5
        self.critic_lr = 0.5
        self.tau = 0.2
        self.capacity = 10000
        self.batch_size = 32

        s_dim = 300
        a_dim = 1

        self.actor = Actor(s_dim, 256, a_dim)
        self.actor_target = Actor(s_dim, 256, a_dim)
        self.critic = Critic(s_dim+a_dim, 256, 1)
        self.critic_target = Critic(s_dim+a_dim, 256, 1)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        self.REWARD=[]
        self.update_i=0

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
        s0, a0, r1, s1,done=transition

        input, hidden = s0
        hx, cx = hidden
        s0 = torch.cat([input, hx, cx], 1)
        s0=s0.detach().numpy()

        input, hidden = s1
        hx, cx = hidden
        s1 = torch.cat([input, hx, cx], 1)
        s1 = s1.detach().numpy()

        self.buffer.append((s0,a0,r1,s1,done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return


        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 ,done= zip(*samples)


        s0 = torch.tensor(s0, dtype=torch.float).squeeze(1)
        a0 = torch.tensor(a0, dtype=torch.float).squeeze(1)
        r1 = torch.tensor(r1, dtype=torch.float).view( self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float).squeeze(1)

        def critic_learn():
            a1 = self.actor_target(s1).squeeze().detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            # action=self.actor(s0)
            # min_reward=100
            # i= random.randint(0, 31)
            # for i in range(32):
            #     xgb = construct_xgb(action.detach().numpy()[:,i,:])
            #     xgb.fit(self.X_train.astype('float') / 256, self.y_train)
            #     pred = xgb.predict(self.X_test)
            #     reward = mean_squared_error(self.y_test, pred)
            #     if reward<min_reward:
            #         min_reward=reward
            # self.REWARD.append(min_reward)
            # print(min_reward)





            loss = -torch.mean( self.critic(s0, self.actor(s0).squeeze()) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        # self.update_i=self.update_i+1
        # if self.update_i%4==0:
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
        # self.update_i=0

    def evaluate(self):
        self.eva_env = env()
        s0 = self.eva_env.reset()
        episode_reward = 0

        for t in range(1, 11):
            # self.env.render()
            input, hidden = s0
            hx, cx = hidden
            s0_temp = torch.cat([input, hx, cx], 1)
            a0 = self.actor_target(s0_temp).squeeze(0).detach().numpy()
            s1, r1, done = self.eva_env.step(a0)
            self.put(s0, a0, r1, s1, done)

            episode_reward += r1
            s0 = s1

            self.learn()

        print('Evaluate Episode  : {}'.format(episode_reward))


    def train(self, num_episodes):
        for i in range(1, num_episodes):
            self.env=env()
            s0 = self.env.reset()
            episode_reward = 0

            for t in range(1, 11):
                # self.env.render()
                a0 = self.act(s0)
                s1, r1, done = self.env.step(a0)
                self.put(s0, a0, r1, s1,done)

                episode_reward += r1
                s0 = s1

                self.learn()

            print('Episode {} : {}'.format(i, episode_reward))
            self.evaluate()


environment=env()
DDPG=Skylark_DDPG()
DDPG.train(100)