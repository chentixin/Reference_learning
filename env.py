import random
import math
import numpy as np
from sympy import diff
import sympy

class Task:
    def __init__(self,data,time):
        self.data = data
        self.max_time = time
        self.current_time = time

class WD:
    def __init__(self, x, y, B_max, T_min, T_max, max_task, W, P,
                 beta, u, sigma2):
        # 新生成任务最大值
        self.B_max = B_max
        # 新生成的任务最大(小）的等待时间
        self.T_max = T_max
        self.T_min = T_min
        # 所有任务的列表
        self.all_task = []
        # 最大任务量
        self.max_task = max_task
        # 信道带宽
        self.W = W
        # 坐标
        self.x = x
        self.y = y
        # 信道功率增益
        self.beta = beta
        # 传输功率
        self.P = P
        # 能量传输效率
        self.u = u
        # 噪声功率
        self.sigma2 = sigma2

    def creat_task(self):
        if len(self.all_task) < self.max_task:
            is_gen = random.random()
            if(is_gen > 0.5):
                t = Task(random.random()*self.B_max, random.randint(self.T_min, self.T_max))
                self.all_task.append(t)

    def del_task(self):
        for each in self.all_task:
            each.current_time -= 1
            if each.current_time == 0:
                self.all_task.pop()

    def offload_task(self, t, at, xy, H):
        if t == 0:
            return 0
        h0 = self.beta/self.get_distance(xy, H)
        power = self.u * self.P * h0 * at
        all_can_off = t * self.W * math.log(1 + (power*h0)/(t*self.sigma2**2), 2)
        now_can_off = float(all_can_off)
        self.all_task.sort(key=lambda x: x.current_time)
        for each in self.all_task:
            if each.data < now_can_off:
                now_can_off -= each.data
                self.all_task.remove(each)

        return all_can_off - now_can_off

    def get_color(self):
        rate = len(self.all_task) * 1.0 / self.max_task
        return [rate*255, 0, (1-rate)*255]

    def get_distance(self, xy, H):
        return math.sqrt(H**2 + (self.x-xy[0])**2 + (self.y-xy[1])**2)

    def get_level_distance(self, xy):
        return math.sqrt((self.x - xy[0]) ** 2 + (self.y - xy[1]) ** 2)

    def get_all_data(self):
        all = 0
        for each in self.all_task:
            all += each.data
        return all

    def get_max_task(self,max_off_task):
        out = []
        for i in range(max_off_task):
            if i < len(self.all_task):
               out.append(self.all_task[i].data)
            else:
                out.append(0)
        return out

class UAV_env:
    def __init__(self, max_x, max_y, T, H, max_angle, V, N_WD, max_off_wd, max_off_task,
                        P, B_max, max_task, T_max, T_min, W, beta, u, sigma2):
        # 无人机飞行区域
        self.max_x = max_x
        self.max_y = max_y
        # 每个时隙的时长
        self.T = T
        # 多少边缘设备
        self.N_WD = N_WD
        # 无人机飞行高度
        self.H = H
        # 通信角度
        self.max_angle = max_angle
        # 无人机初始位置（随机生成）
        self.U_x = random.uniform(0, max_x)
        self.U_y = random.uniform(0, max_y)
        # 无人机飞行速度
        self.V = V
        # 最大选择卸载节点个数，最大单个节点卸载任务个数
        self.max_off_wd = max_off_wd
        self.max_off_task = max_off_task
        # 随机生成WD设备
        self.WDs = []
        self.B_max = B_max
        self.max_task = max_task
        self.T_max = T_max
        self.T_min = T_min
        self.W = W
        self.P = P
        self.beta = beta
        self.u = u
        self.sigma2 = sigma2
        for i in range(N_WD):
            x = random.uniform(0, max_x)
            y = random.uniform(0, max_y)
            new = WD(x=x, y=y, B_max=B_max, max_task=max_task,T_max=T_max,T_min=T_min, W=W, P=P, beta=beta, u=u, sigma2=sigma2)
            self.WDs.append(new)


    def get_state(self):
        state = [self.U_x,self.U_y]
        for each in self.WDs:
            state.append(each.x)
            state.append(each.y)
            state.append(len(each.all_task))
        return np.array(state, dtype=float)

    def step(self, fly_action, task_action, off_number):
        all_off_data = 0
        for i,index in enumerate(off_number):
            all_off_data += self.WDs[index].offload_task(task_action[i+1], task_action[0], [self.U_x, self.U_y],self.H)
        task_reward = float(all_off_data)

        self.U_x += math.cos(fly_action) * self.V
        self.U_y += math.sin(fly_action) * self.V

        all_env_data = 0
        for each in self.WDs:
            all_env_data += each.get_all_data()

        max_r = self.H * math.tan(self.max_angle)
        is_ok = lambda x : x.get_level_distance([self.U_x, self.U_y]) <= max_r
        indices = [index for index, value in enumerate(self.WDs) if is_ok(value)]

        all_in_data = 0
        for index in indices:
            all_in_data += self.WDs[index].get_all_data()

        fly_next_state = []
        fly_next_state.append(self.U_x)
        fly_next_state.append(self.U_y)
        for each in self.WDs:
            # fly_next_state += [each.x,each.y]
            fly_next_state.append(each.get_all_data()/10)

        if all_in_data != 0:
            fly_reward = all_in_data * 1.0 / all_env_data
        else:
            fly_reward = 0

        if self.U_x < 0 or self.U_x > self.max_x or self.U_y < 0 or self.U_y > self.max_y:
            fly_reward *= 0.9
            
        return np.array(fly_next_state, dtype=float), fly_reward

        # 在开头生成任务，更新环境

    def update_env(self, off_number):
        for each in self.WDs:
            each.del_task()
            each.creat_task()

        state_task = []
        for i in range(self.max_off_wd):
            if i < len(off_number):
                state_task += [self.WDs[off_number[i]].get_distance([self.U_x, self.U_y], self.H)]
                state_task += self.WDs[off_number[i]].get_max_task(self.max_off_task)
            else:
                state_task += [0 for i in range(self.max_off_task + 1)]

        state_fly = [self.U_x, self.U_y]
        for each in self.WDs:
            state_fly += [each.x, each.y]
            state_fly.append(each.get_all_data()/10)

        return state_task, state_fly

    def restart(self):
        state1 = []
        state2 = []
        # 无人机初始位置（随机生成）
        self.U_x = 0
        self.U_y = 0
        state2.append(self.U_x)
        state2.append(self.U_y)

        for i in range(5):
            each = self.WDs[i]
            state1 += [each.get_distance([self.U_x, self.U_y], self.H)]
            state1 += [0 for i in range(self.max_off_task)]

        for each in self.WDs:
            each.all_task = []
            state2 += [each.x, each.y]
            state2.append(0)

        return np.array(state2, dtype=float)

    def random_action(self):
        return random.uniform(0, 2*math.pi)

    def save_state(self,path):
        with open(path, 'w') as file:
            for each in self.WDs:
                file.write(f"{each.x}\n")
                file.write(f"{each.y}\n")

if __name__ == "__main__":
    temp = WD(1,1,100,1000,1e6,0.1,1e-4,0.2,1e-5)
    print(temp.get_FPD_t(2,[4,5],10,0.2))