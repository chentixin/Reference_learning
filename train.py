import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from parsers import args
from DDPG import ReplayBuffer, DDPG
from Fly_DDPG import Fly_DDPG
from Task_DDPG import Task_DDPG
from env import UAV_env
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 用于环境绘制
# ----------------------------------------------------------
# ----------------------------------------------------------
if args.is_show:
    # 创建图形和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 初始化为空的点集
    scat = ax.scatter([], [], [], c=[])

    # 设置轴的范围
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 15)

    plt.ion()  # 打开交互模式
    plt.show()

def get_color(all_task, max_task):
    rate = all_task * 1.0 / max_task
    if rate > 1 :
        print(all_task)
        print(max_task)
    return [rate, 0, (1 - rate)]

# 更新函数，用于实时接收数据并更新图像
def update(state,H,num):
    points = []
    colors = []
    points.append([state[0],state[1],H])
    colors.append([0,1,0])
    for i in range(num):
        points.append([state[i*3 + 2], state[i*3 + 3], 0])
        colors.append(get_color(state[i*3 + 4], args.max_task))
    # 清除当前散点并绘制新的点
    ax.cla()  # 清空原有内容
    ax.set_xlim(0, 100)  # 重新设定轴范围
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 15)

    points = np.array(points)
    colors = np.array(colors)
    # 绘制新的点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)

    # 画园
    theta = np.linspace(0, 2 * np.pi, 100)  # 生成0到2π之间的100个点
    x_circle = state[0] + 20 * np.cos(theta)
    y_circle = state[1] + 20 * np.sin(theta)
    z_circle = np.full_like(x_circle, 0)  # 圆在平面 z = center[2] 上
    ax.plot(x_circle, y_circle, z_circle, color='g')


    plt.pause(0.01)
    plt.draw()
# -----------------------------------------------------------------------
# -------------------------------------- #
# 绘图
# -------------------------------------- #
def draw_lens(return_list, mean_return_list):
    x_range = list(range(len(return_list)))
    plt.subplot(121)
    plt.plot(x_range, return_list)  # 每个回合return
    plt.xlabel('episode')
    plt.ylabel('return')
    plt.subplot(122)
    plt.plot(x_range, mean_return_list)  # 每回合return均值
    plt.xlabel('episode')
    plt.ylabel('mean_return')
    plt.savefig('sine_wave.png', format='png', dpi=300)

# ----------------------------------------------------------
# ----------------------------------------------------------

env = UAV_env(max_x = args.max_x, max_y= args.max_y,
              T=args.T, H=args.H, max_angle=args.max_angle,
              V=args.V, N_WD=args.N_WD,
              max_off_wd=5, max_off_task=5,
              P=args.P,B_max=args.B_max,
              max_task=args.max_task, W=args.W,
              beta=args.beta, u=args.u, sigma2=args.sigma2,
              T_max=args.T_max,T_min=args.T_min)

# 参数初始化
fly_n_states = args.N_WD * 3 + 2
fly_n_actions = 1
fly_action_bound = math.pi

task_n_states = args.task_max_off_wd * (args.task_max_off_task + 1)
task_n_actions = args.task_max_off_wd + 1

# 经验回放池实例化
fly_replay_buffer = ReplayBuffer(capacity=args.fly_buffer_size)
task_replay_buffer = ReplayBuffer(capacity=args.task_buffer_size)
# 模型实例化
fly_agent = Fly_DDPG(n_states=fly_n_states,  # 状态数
             n_hiddens=args.fly_n_hiddens,  # 隐含层数
             n_actions=fly_n_actions,  # 动作数
             action_bound=fly_action_bound,  # 动作最大值
             sigma=args.fly_sigma,  # 高斯噪声
             actor_lr=args.fly_actor_lr,  # 策略网络学习率
             critic_lr=args.fly_critic_lr,  # 价值网络学习率
             tau=args.fly_tau,  # 软更新系数
             gamma=args.fly_gamma,  # 折扣因子
             device=device
             )

task_agent = Task_DDPG(n_states=task_n_states,  # 状态数
             n_hiddens=args.task_n_hiddens,  # 隐含层数
             n_actions=task_n_actions,  # 动作数
             sigma=args.fly_sigma,  # 高斯噪声
             actor_lr=args.fly_actor_lr,  # 策略网络学习率
             critic_lr=args.fly_critic_lr,  # 价值网络学习率
             tau=args.fly_tau,  # 软更新系数
             gamma=args.fly_gamma,  # 折扣因子
             device=device
             )

# -------------------------------------- #
# 模型训练
# -------------------------------------- #

return_list = []  # 记录每个回合的return
mean_return_list = []  # 记录每个回合的return均值
num_start = 0
num_epochs = 10000

# # 进行断点重训
# if args.is_continue:
#     print("continue train")
#     agent.actor.load_state_dict(torch.load("action.pt"))
#     agent.critic.load_state_dict(torch.load("critic.pt"))
#     # 加载环境
#     with open("state.txt", 'r') as file:
#         wd_id = 0
#         for line in file:
#             env.WDs[wd_id].x = (float(line.strip()))
#             line = file.readline()
#             env.WDs[wd_id].y = (float(line.strip()))
#             wd_id += 1
#
#     # 加载学习率
#     with open("parsers.txt", 'r') as file:
#         line = file.readline()
#         num_start = (int(line.strip()))
#         line = file.readline()
#         args.actor_lr = (float(line.strip()))
#         line = file.readline()
#         args.critic_lr = (float(line.strip()))
#         print(args.actor_lr, args.critic_lr)
# elif args.is_use:
#     print("use last pt")
#     agent.actor.load_state_dict(torch.load("action.pt"))
#     agent.critic.load_state_dict(torch.load("critic.pt"))
# else:
#     # 记录本次环境信息
#     env.save_state("state.txt")

try:
    for i in range(num_start, num_epochs):  # 迭代10回合
        episode_return = 0  # 累计每条链上的reward
        state = env.restart()  # 初始时的状态
        done = False  # 回合结束标记

        # 学习率递减
        if i <= 2000:
            args.actor_lr *= 0.9993
            args.critic_lr *= 0.9993
        else:
            args.actor_lr *= 0.9998
            args.critic_lr *= 0.9998

        fly_agent.change_lr(args.fly_actor_lr, args.fly_critic_lr)
        task_agent.change_lr(args.task_actor_lr, args.task_critic_lr)

        loop = tqdm(range(60))
        for j in loop:
            state_task, state_fly = env.update_env([])
            # 获取当前状态对应的动作
            fly_action = fly_agent.take_action(state)
            task_action = task_agent.take_action(state)

            # 随机探索（逐渐减弱）若加载已经训练过的模型则不随机
            if i < 2000 and not args.is_continue:
                fly_action += np.random.normal(loc=0, scale=0.1 * np.cos((i * math.pi) / (2 * 2000)))

            # 环境更新
            next_state, reward, done = env.step(fly_action, task_action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward

            # 更新三维图
            if args.is_show and i % 20 == 0:
                update(env.get_state(), H=args.H, num=args.N_WD)

            # 如果经验池超过容量，开始训练
            if replay_buffer.size() >= args.min_size:
                s, a, r, ns, d = replay_buffer.sample(args.batch_size)
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'rewards': r,
                    'next_states': ns,
                    'dones': d,
                }
                agent.update(transition_dict)

            loop.set_description(f'Epoch [{i}/{num_epochs}]')
            loop.set_postfix(reward=episode_return)

        # 保存每一个回合的回报
        return_list.append(episode_return)
        mean_return_list.append(np.mean(return_list[-10:]))  # 平滑

        # 保存训练的数据
        with open("parsers.txt", 'w') as file:
            file.write(str(i) + "\n")
            file.write(f"{args.actor_lr}\n")
            file.write(f"{args.critic_lr}\n")
except:
    print("have error")
    draw_lens(return_list, mean_return_list)
    raise

# 保存此次训练的数据
# torch.save(agent.actor.state_dict(),"action.pt")
# torch.save(agent.critic.state_dict(), "critic.pt")
draw_lens(return_list,mean_return_list)