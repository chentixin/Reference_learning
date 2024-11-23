import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random


# 策略（动作）模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(2 + number_of_devices, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
            torch.nn.Tanh(),
        )

    def forward(self, state):
        return self.sequential(state.to(device))


model_action = Model().to(device)
model_action_next = Model().to(device)
model_action_next.load_state_dict(model_action.state_dict())


# 价值模型
# model_value = torch.nn.Sequential(
#     torch.nn.Linear(2 + number_of_devices + 1, neuron_middle_layer),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neuron_middle_layer, neuron_middle_layer),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neuron_middle_layer, neuron_middle_layer),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neuron_middle_layer, 1),
# )
# model_value_next = torch.nn.Sequential(
#     torch.nn.Linear(2 + number_of_devices + 1, neuron_middle_layer),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neuron_middle_layer, neuron_middle_layer),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neuron_middle_layer, neuron_middle_layer),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neuron_middle_layer, 1),
# )

# 价值模型
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(2 + number_of_devices + 1, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.sequential(state.to(device))


model_value = Model2().to(device)
model_value_next = Model2().to(device)
model_value_next.load_state_dict(model_value.state_dict())


# 输入状态，根据动作模型得到动作
def get_action(state, action_explore):
    # state =
    action = model_action(state.float().to(device))
    # action_explore = 0.5
    # 给动作添加噪声,增加探索，改变这两个非常显著影响收敛速度
    action += random.normalvariate(mu=0, sigma=action_explore)

    return action


# 构造离线样本池
datas = []


# 向样本池中添加N条数据,删除M条最古老的数据
def update_data(action_explore):
    N = 0
    state = env.reset()
    while N < 100:
        # 根据当前状态得到动作
        action = get_action(state, action_explore)
        # 执行动作,得到反馈
        nextState, reward = env.step(state, action)
        # 记录数据样本
        datas.append((state, action, reward, nextState))
        state = nextState
        N += 1

    # 数据上限,超出时从最古老的开始删除
    while len(datas) > 10000:
        datas.pop(0)


def get_sample():
    # 从样本池中采样
    samples = random.sample(datas, 100)

    # #[b, 12]
    state = torch.FloatTensor(np.array([i[0].cpu().detach().numpy() for i in samples])).reshape(-1,
                                                                                                2 + number_of_devices).to(
        device)
    # [b, 2]
    action = torch.FloatTensor(np.array([i[1].cpu().detach().numpy() for i in samples])).reshape(-1, 1).to(device)
    # #[b, 1]
    reward = torch.FloatTensor(np.array([i[2].cpu().detach().numpy() for i in samples])).reshape(-1, 1).to(device)
    # [b, 12]
    next_state = torch.FloatTensor(np.array([i[3].cpu().detach().numpy() for i in samples])).reshape(-1,
                                                                                                     2 + number_of_devices).to(
        device)

    return state, action, reward, next_state


def get_value(state, action):
    # 直接评估综合了state和action的value
    # [b,12] -> [b,14]
    input = torch.cat([state.to(device), action.to(device)], dim=1)  # 按行拼接起来，把action加在后面

    return model_value(input)


def get_target(next_state, reward):
    # 对next_state的评估需要先把它对应的动作计算出来,这里用model_action_next来计算
    # [b, 12] -> [b, 2]
    action = model_action_next(next_state.to(device))
    # 和value的计算一样,action拼合进next_state里综合计算
    # [b, 12+1] -> [b, 13]
    input = torch.cat([next_state, action], dim=1).to(device)
    # 评估动作的目标价值大小
    # [b, 14] -> [b, 1]
    target = model_value_next(input) * 0.98
    # [b, 1] + [b, 1] -> [b, 1]
    target += reward

    return target


def get_loss_action(state):
    # 首先把动作计算出来
    # [b, 12] -> [b, 1]
    action = model_action(state)
    # 像value计算那里一样,拼合state和action综合计算
    # [b, 3+1] -> [b, 4]
    input = torch.cat([state, action], dim=1).to(device)
    # 使用value网络评估动作的价值,价值是越高越好
    # 因为这里是在计算loss,loss是越小越好,所以符号取反
    # [b, 4] -> [b, 1] -> [1]
    loss = -model_value(input).mean()

    return loss


def soft_update(model, model_next):
    for param, param_next in zip(model.parameters(), model_next.parameters()):
        # 以一个小的比例更新
        value = param_next.data * 0.995 + param.data * 0.005
        param_next.data.copy_(value)


M = 2000  # M批数据，M个迭代次数，或者进度条，M能提升数据扩展的广度
N = 100  # 每批数据迭代N次，总计产生 M * N个计算结果，N能提升数据学习的成熟度

num_episodes = M * N


def train():
    model_action.train()
    model_value.train()
    lr_actor = 0.0001
    lr_critic = 0.0001
    optimizer_action = torch.optim.Adam(model_action.parameters(), lr=lr_actor)
    optimizer_value = torch.optim.Adam(model_value.parameters(), lr=lr_critic)
    loss_fn = torch.nn.MSELoss()
    reward_list = []
    reward_list1 = []
    loss_list = []
    action_list = []
    loss_action_list = []
    for i in range(M):  # 200个进度条
        # 更新N条数据
        if i < 2000:
            action_explore = 0.1 * np.cos(i / 1500 * np.pi / 2)  # 探索逐渐变小
            # update_data(action_explore)
            lr_actor *= 0.9993
            lr_critic *= 0.9993
        else:
            action_explore = 0  # 后续无探索
            lr_actor *= 0.9998
            lr_critic *= 0.9998

        update_data(action_explore)

        with tqdm(total=int(N), desc='迭代次数： %d' % i) as pbar:
            for i_episode in range(int(N)):
                # 采样一批数据，输出的是tensor
                state, action, reward, next_state = get_sample()
                action_list.append(action.mean())

                # 计算value和target
                value = get_value(state, action)  # 传入的是tensor
                reward_mean = reward.mean()
                target = get_target(next_state, reward)  # 传入list，输出tensor

                # 两者求差,计算loss,用价值和目标价值来更新价值模型的参数td_error
                loss_value = loss_fn(value, target)

                optimizer_value.lr = lr_critic
                optimizer_value.zero_grad()
                loss_value.backward()
                optimizer_value.step()

                # 把state和action输入到价值网络中得到差距，并对差距求平均后得到loss，用loss更新动作模型
                loss_action = get_loss_action(state)
                loss_action_list.append(loss_action)

                optimizer_action.lr = lr_actor
                optimizer_action.zero_grad()
                loss_action.backward()
                optimizer_action.step()

                # 以一个小的比例更新
                soft_update(model_action, model_action_next)
                soft_update(model_value, model_value_next)

                reward_list.append(reward_mean.cpu().detach().numpy())
                loss_list.append(loss_value.cpu().mean().detach().numpy())

                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix(
                        {'episode': '%d' % (M * i + i_episode + 1), 'reward': '%.3f' % np.mean(reward_list[-100:]),
                         'loss': '%.3f' % np.mean(loss_list[-100:])})
                    reward_list1.append(np.mean(reward_list[-100:]))
                pbar.update(1)

    episode_list = list(range(len(reward_list)))
    plt.plot(episode_list, reward_list)
    plt.title("reward")
    # plt.show()

    plt.plot(episode_list, loss_list)
    plt.title("loss")
    # plt.show()

    return reward_list, episode_list, loss_list, reward_list1