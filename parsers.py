# 参数定义
import argparse  # 参数设置

# 创建解释器
parser = argparse.ArgumentParser()

parser.add_argument('--max_x', type=int, default=100, help='最大x')
parser.add_argument('--max_y', type=int, default=100, help='最大y')
parser.add_argument('--T', type=int, default=1, help='每个时隙时间')
parser.add_argument('--T_max', type=int, default=10, help='每个时隙时间')
parser.add_argument('--T_min', type=int, default=5, help='每个时隙时间')
parser.add_argument('--H', type=int, default=10, help='无人机高度')
parser.add_argument('--max_angle', type=int, default=20, help='可通信最大半径')
parser.add_argument('--V', type=int, default=10, help='无人机飞行速度')
parser.add_argument('--N_WD', type=int, default=20, help='wd的个数')
parser.add_argument('--P', type=int, default=0.05, help='能量传输功率')
parser.add_argument('--B_max', type=int, default=1000, help='每隔时隙生成最大任务')
parser.add_argument('--max_task', type=int, default=20, help='最大任务量')
parser.add_argument('--W', type=int, default=1e6, help='信道带宽')
parser.add_argument('--beta', type=int, default=1e-4, help='信道增益')
parser.add_argument('--u', type=int, default=0.2, help='能量传输效率')
parser.add_argument('--sigma2', type=int, default=1e-5, help='噪声')

parser.add_argument('--is_use', type=int, default=False, help='是否使用以前的权重')
parser.add_argument('--is_continue', type=int, default=False, help='是否断点重训')
parser.add_argument('--is_show', type=int, default=True, help='是否展示无人机动态图')

# 网络参数定义
# 无人机飞行网络
parser.add_argument('--fly_actor_lr', type=float, default=1e-5, help='策略网络的学习率')
parser.add_argument('--fly_critic_lr', type=float, default=5e-4, help='价值网络的学习率')
parser.add_argument('--fly_n_hiddens', type=int, default=512, help='隐含层神经元个数')
parser.add_argument('--fly_gamma', type=float, default=0.98, help='折扣因子')
parser.add_argument('--fly_tau', type=float, default=0.05, help='软更新系数')
parser.add_argument('--fly_buffer_size', type=int, default=5000, help='经验池容量')
parser.add_argument('--fly_min_size', type=int, default=60, help='经验池超过120再训练')
parser.add_argument('--fly_batch_size', type=int, default=60, help='每次训练多少组样本')
parser.add_argument('--fly_sigma', type=int, default=0.01, help='高斯噪声标准差')

# 任务卸载网络
parser.add_argument('--task_actor_lr', type=float, default=1e-5, help='策略网络的学习率')
parser.add_argument('--task_critic_lr', type=float, default=5e-4, help='价值网络的学习率')
parser.add_argument('--task_n_hiddens', type=int, default=512, help='隐含层神经元个数')
parser.add_argument('--task_gamma', type=float, default=0.98, help='折扣因子')
parser.add_argument('--task_tau', type=float, default=0.05, help='软更新系数')
parser.add_argument('--task_buffer_size', type=int, default=5000, help='经验池容量')
parser.add_argument('--task_min_size', type=int, default=60, help='经验池超过120再训练')
parser.add_argument('--task_batch_size', type=int, default=60, help='每次训练多少组样本')
parser.add_argument('--task_sigma', type=int, default=0.01, help='高斯噪声标准差')
parser.add_argument('--task_max', type=int, default=60, help='每次训练多少组样本')
parser.add_argument('--task_max_off_wd', type=int, default=5, help='最大可以卸载的节点个数')
parser.add_argument('--task_max_off_task', type=int, default=5, help='每个节点传入网络的参数')

# 参数解析
args=parser.parse_args()