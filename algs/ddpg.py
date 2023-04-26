import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from algs.util.replay_buffer import ReplayBuffer

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, train=True) -> None:
        # the action bound and state_dim here are dicts
        super().__init__()
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.train = train
        # self.LaneEncoder = LaneEncoder()
        # self.layer_norm=nn.LayerNorm(128)
        # self.batch_norm=nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.2)

        self.fc1_1 = nn.Linear(state_dim['waypoints'], 64)
        self.fc1_2 = nn.Linear(state_dim['ego_vehicle'],32)
        self.fc1_3 = nn.Linear(state_dim['companion_vehicle'], 32)
        # concat the first layer output and input to second layer
        self.fc2 = nn.Linear(128,128)
        self.fc_out = nn.Linear(128, action_dim)

        # torch.nn.init.normal_(self.fc1_1.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc1_2.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.xavier_normal_(self.fc1_1.weight.data)
        # torch.nn.init.xavier_normal_(self.fc1_2.weight.data)
        # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

    def forward(self, state):
        # state : waypoints info+ companion_vehicle info, shape: batch_size*22, first 20 elements are waypoints info,
        # the rest are vehicle info
        state_wp = state[:, :self.state_dim['waypoints']]
        state_ev = state[:,-self.state_dim['companion_vehicle']-self.state_dim['ego_vehicle']:-self.state_dim['companion_vehicle']]
        state_vf = state[:, -self.state_dim['companion_vehicle']:]
        state_wp = F.relu(self.fc1_1(state_wp))
        state_ev=F.relu((self.fc1_2(state_ev)))
        state_vf = F.relu(self.fc1_3(state_vf))
        state_ = torch.cat((state_wp,state_ev, state_vf), dim=1)
        hidden = F.relu(self.fc2(state_))
        action = torch.tanh(self.fc_out(hidden))
        # steer,throttle_brake=torch.split(out,split_size_or_sections=[1,1],dim=1)
        # steer=steer.clone()
        # throttle_brake=throttle_brake.clone()
        # steer*=self.action_bound['steer']
        # throttle=throttle_brake.clone()
        # brake=throttle_brake.clone()
        # for i in range(throttle.shape[0]):
        #     if throttle[i][0]<0:
        #         throttle[i][0]=0
        #     if brake[i][0]>0:
        #         brake[i][0]=0
        # throttle*=self.action_bound['throttle']
        # brake*=self.action_bound['brake']

        return action


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        # parameter state_dim here is a dict
        super().__init__()

        #self.state_dim = state_dim['waypoints'] + state_dim['ego_vehicle']+state_dim['companion_vehicle']
        self.state_dim=state_dim

        self.action_dim = action_dim
        self.layer_norm = nn.LayerNorm(128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        #self.fc1 = nn.Linear(self.state_dim + action_dim, 64)

        self.fc1_1=nn.Linear(self.state_dim['waypoints'],32)
        self.fc1_2=nn.Linear(self.state_dim['ego_vehicle'],32)
        self.fc1_3=nn.Linear(self.state_dim['companion_vehicle'],32)
        self.fc1_4=nn.Linear(self.action_dim,32)
        self.fc2=nn.Linear(128,128)
        self.fc_out = nn.Linear(128, 1)

        # torch.nn.init.normal_(self.fc1.weight.data,0,0.01)
        # torch.nn.init.normal_(self.fc_out.weight.data,0,0.01)
        # torch.nn.init.xavier_normal_(self.fc1.weight.data)
        # torch.nn.init.xavier_normal_(self.fc_out.weight.data)

    def forward(self, state, action):

        # state : waypoints info+ companion_vehicle info, shape: batch_size*22, first 20 elements are waypoints info,
        # the rest are vehicle info
        state_wp = state[:, :self.state_dim['waypoints']]
        state_ev = state[:,-self.state_dim['companion_vehicle']-self.state_dim['ego_vehicle']:-self.state_dim['companion_vehicle']]
        state_vf = state[:, -self.state_dim['companion_vehicle']:]
        state_wp=F.relu(self.fc1_1(state_wp))
        state_ev=F.relu(self.fc1_2(state_ev))
        state_vf=F.relu(self.fc1_3(state_vf))
        state_ac=F.relu(self.fc1_4(action))
        state = torch.cat((state_wp,state_ev,state_vf, state_ac), dim=1)
        hidden=F.relu(self.fc2(state))
        out = self.fc_out(hidden)

        return out

class MLP(nn.Module):
    r"""
    Construct a MLP in LaneEncoder, include a single fully-connected layer,
    followed by layer normalization and then ReLU.
    """

    def __init__(self, input_size, hidden_size=64):
        r"""
        self.norm is layer normalization.
        Args:
            input_size: the size of input layer.
            hidden_size: the size of output layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r"""
        Args:
            x: x.shape = [batch_size, n, input_size]
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        # x = self.fc2(x)
        return x


class LaneEncoder(torch.nn.Module):
    def __init__(self, waypoint_dim, hidden_size):
        super(LaneEncoder, self).__init__()
        self.waypoint_dim = waypoint_dim
        self.hidden_size = hidden_size
        self.MLP = MLP(self.waypoint_dim, self.hidden_size)

    def forward(self, waypoints):
        """
        :param waypoints: [batch_size, n, input_size]
        :return: (batch_size, n, input_size*2)
        """
        x = self.MLP(waypoints)
        batch_size, n, input_size = x.shape
        x2 = x.permute(0, 2, 1)  # [batch_size, input_size, n]
        x2 = F.max_pool1d(x2, kernel_size=x2.shape[2])  # [batch_size, input_size, 1]
        x2 = torch.cat([x2]*n, dim=2)  # [batch_size, input_size, n]
        y = torch.cat((x2.permute(0, 2, 1), x), dim=2)  # [batch_size, n, input_size*2]
        return y


class DDPG:
    def __init__(self, state_dim, action_dim, action_bound, gamma, tau, sigma, theta, epsilon,
                 buffer_size, batch_size, actor_lr, critic_lr, device) -> None:
        self.learn_time = 0
        self.replace_a = 0
        self.replace_c = 0
        self.s_dim = state_dim  # state_dim here is a dict
        self.s_dim['waypoints']*=2  # The input waypoints info has been compressed
        self.a_dim, self.a_bound = action_dim, action_bound
        self.theta = theta
        self.gamma, self.tau, self.sigma, self.epsilon = gamma, tau, sigma, epsilon  # sigma:高斯噪声的标准差，均值直接设置为0
        self.buffer_size, self.batch_size, self.device = buffer_size, batch_size, device
        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        # adjust different types of replay buffer
        #self.replay_buffer = Split_ReplayBuffer(buffer_size)
        self.replay_buffer =ReplayBuffer(buffer_size)
        # self.replay_buffer = offline_replay_buffer()
        """self.memory=torch.tensor((buffer_size,self.s_dim*2+self.a_dim+1+1),
            dtype=torch.float32).to(self.device)"""
        self.pointer = 0  # serve as updating the memory data
        self.train = True

        self.actor = PolicyNet(self.s_dim, self.a_dim, self.a_bound).to(self.device)
        self.actor_target = PolicyNet(self.s_dim, self.a_dim, self.a_bound).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = QValueNet(self.s_dim, self.a_dim).to(self.device)
        self.critic_target = QValueNet(self.s_dim, self.a_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.loss = nn.MSELoss()

        self.steer_noise = OrnsteinUhlenbeckActionNoise(self.sigma, self.theta)
        self.tb_noise = OrnsteinUhlenbeckActionNoise(self.sigma, self.theta)

    def take_action(self, state):
        state_wps = torch.tensor(state['waypoints'], dtype=torch.float32).view(1, -1).to(self.device)
        state_ev=torch.tensor(state['ego_vehicle'],dtype=torch.float32).view(1,-1).to(self.device)
        state_vf = torch.tensor(state['vehicle_front'], dtype=torch.float32).view(1, -1).to(self.device)
        state_ = torch.cat((state_wps,state_ev, state_vf), dim=1)
        # print(state_.shape)
        action = self.actor(state_)
        
        if self.a_dim == 3:
            if (action[0, 0].is_cuda):
                action = np.array([action[:, 0].detach().cpu().numpy(), action[:, 1].detach().cpu().numpy(), action[:, 2].detach().cpu().numpy()]).reshape((-1, 3))
            else:
                action = np.array([action[:, 0].detach().numpy(), action[:, 1].detach().numpy(), action[:, 2].detach().numpy()]).reshape((-1, 3))
            print(f"Network Output - Steer: {action[0][0]}, Throttle_brake: {action[0][1]}, Exec_steps:{action[0][2]}")
        else:
            if (action[0, 0].is_cuda):
                action = np.array([action[:, 0].detach().cpu().numpy(), action[:, 1].detach().cpu().numpy()]).reshape((-1, 2))
            else:
                action = np.array([action[:, 0].detach().numpy(), action[:, 1].detach().numpy()]).reshape((-1, 2))
            print(f"Network Output - Steer: {action[0][0]}, Throttle_brake: {action[0][1]}")
        # if np.random.random()<self.epsilon:
        if self.train:
            action[:, 0] = np.clip(np.random.normal(action[:, 0], self.sigma), -1, 1)
            action[:, 1] = np.clip(np.random.normal(action[:, 1], self.sigma), -1, 1)
            if self.a_dim == 3:
                action[:, 2] = np.clip(np.random.normal(action[:, 2], self.sigma), -1, 1)
                print(f'After noise - Steer: {action[0][0]}, Throttle_brake: {action[0][1]}, Exec_steps:{action[0][2]}')
            else:
                print(f'After noise - Steer: {action[0][0]}, Throttle_brake: {action[0][1]}')
        # if self.train:
        #     action[:,0]=np.clip(action[:,0]+self.steer_noise(),-1,1)
        #     action[:,1]=np.clip(action[:,1]+self.tb_noise(),-1,1)
        #     action[:,2]=np.clip(action[:,2]+self.tb_noise(),-1,1)
        
        return action


    def learn(self):
        self.learn_time += 1
        # if self.learn_time > 100000:
        #     self.train = False
        self.replace_a += 1
        self.replace_c += 1
        b_s, b_a, b_r, b_ns, b_t, b_d, b_i = self.replay_buffer.sample(self.batch_size)
        # 此处得到的batch是否是pytorch.tensor?
        batch_s = torch.tensor(b_s, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_ns = torch.tensor(b_ns, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_a = torch.tensor(b_a, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_r = torch.tensor(b_r, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_d = torch.tensor(b_d, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        batch_t = torch.tensor(b_t, dtype=torch.float32).view((self.batch_size, -1)).to(self.device)
        # compute the target Q value using the information of next state
        action_target = self.actor_target(batch_ns)
        next_q_values = self.critic_target(batch_ns, action_target)
        q_targets = batch_r + self.gamma * next_q_values * (1 - batch_t)
        critic_loss = self.loss(self.critic(batch_s, batch_a), q_targets)
        print(f'TD-error:{critic_loss}')
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # print()
        # self._print_grad(self.critic)
        # print()
        self.critic_optimizer.step()

        action = self.actor(batch_s)
        q = self.critic(batch_s, action)
        actor_loss = -torch.mean(q)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def _print_grad(self, model):
        '''Print the grad of each layer'''
        for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.steer_noise.set_sigma(sigma)
        self.tb_noise.set_sigma(sigma)

    def reset_noise(self):
        self.steer_noise.reset()
        self.tb_noise.reset()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, net, target_net):
        net.load_state_dict(target_net.state_dict())

    def store_transition(self, state, action, reward, next_state, truncated, done,info):  # how to store the episodic data to buffer
        def _compress(state):
            """return state : waypoints info+ vehicle_front info, shape: 1*22, 
            first 20 elements are waypoints info, the rest are vehicle info"""
            wps = np.array(state['waypoints'], dtype=np.float32).reshape((1, -1))
            ev = np.array(state['ego_vehicle'],dtype=np.float32).reshape((1,-1))
            vf = np.array(state['vehicle_front'], dtype=np.float32).reshape((1, -1))
            state_ = np.concatenate((wps, ev, vf), axis=1)

            return state_
        
        state=_compress(state)
        next_state=_compress(next_state)
        self.replay_buffer.add((state, action, reward, next_state, truncated, done,info))
        return

    def save_net(self,file='./out/ddpg_final.pth'):
        state = {
            'actor': self.actor.state_dict(),
            'actor_target':self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target':self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(state, file)

    def load_net(self, state):
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, sigma, theta=0.001, mu=np.array([0.0]), dt=1e-2, x0=None):
        """
        mu: The mean value of action
        theta: The bigger the value, the faster noise get close to mu
        sigma: Noise amplifier, the bigger the value, the more amplification of the disturbance
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
