import yaml
import numpy as np
import math
from math import sin, cos, atan2 as arctan

from .mode import Trajectory,UAV

# 1.按照指定时间返回飞行信息
# 2.自行迭代返回飞行信息（迭代器）
class Generator():
    def __init__(self,config_path):
        self.load_config(config_path)
        self.g = np.array([0,0,9.8])

    def load_config(self,config_path):
        f = open(config_path,"r",encoding="UTF-8")
        configure = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        self.trajectory_info = Trajectory(configure["trajectory"]) # 读取轨迹信息
        self.UAV_info = UAV(configure["uav"]) # 读取无人机信息
        self.clock = self.trajectory_info.start_time # 时钟

        self.offsets = np.array(self.UAV_info.offsets) # 无人机偏移量
        return

    def clock_update(self):
        self.clock += self.trajectory_info.delta_t

    # 获取无人机的姿态
    def get_attitude_by_F(self,F):
        psi = 0 # 无人机的航向角
        theta_a = arctan(F[0] * cos(psi) + F[1] * sin(psi), F[2]) # 无人机的俯仰角
        phi_a = arctan(F[0] * sin(psi) - F[1] * cos(psi), # 无人机的滚转角
                       F[2] / cos(theta_a))
        attitude = np.array([phi_a, theta_a, psi])
        return attitude

    # output:[4,3]
    # uav_i:无人机编号 t:时间 output:位置、速度、加速度、姿态
    def get_single_state_by_t(self,uav_i,t):
        p = np.array([
            self.trajectory_info.radius * sin(2 * math.pi * t / self.trajectory_info.period),
            self.trajectory_info.radius * cos(2 * math.pi * t / self.trajectory_info.period),
            0
        ]) # 位置

        p_dot = np.array([
                2 * math.pi / self.trajectory_info.period * self.trajectory_info.radius * cos(2 * math.pi * t / self.trajectory_info.period),
                -2 * math.pi / self.trajectory_info.period * self.trajectory_info.radius * sin(2 * math.pi * t / self.trajectory_info.period),
                0
        ]) # 速度

        p_d_dot2 = np.array([
                -(2 * math.pi / self.trajectory_info.period) ** 2 * self.trajectory_info.radius * sin(2 * math.pi * t / self.trajectory_info.period),
                -(2 * math.pi / self.trajectory_info.period) ** 2 * self.trajectory_info.radius * cos(2 * math.pi * t / self.trajectory_info.period),
                0
        ])# 加速度

        F = self.UAV_info.uav_weight * (self.g + p_d_dot2) # 无人机的受力
        attitude = self.get_attitude_by_F(F) # 无人机的姿态
        return p+self.offsets[uav_i], p_dot, p_d_dot2, attitude

    # output:[5,13]
    # 获取所有无人机的状态
    def get_swarm_state_by_t(self,t):
        swarm_state = []
        for uav_i in range(self.UAV_info.uav_num):
            p,p_dot,p_dot2,att = self.get_single_state_by_t(uav_i,t)
            swarm_state.append(np.concatenate([np.array([t]),p,p_dot,p_dot2,att,p],axis=0)) # p重复？
        # print(swarm_state)
        return np.stack(swarm_state,axis=0)

    # output:[start_point_num,5,13]
    def get_swarm_state_by_clock(self):
        return self.get_swarm_state_by_t(self.clock)

    # output:[12]
    # 获取单个无人机的状态
    def get_single_state_by_command(self,pos_cur,vel_cur,pos_nxt): # pos_cur:当前位置 vel_cur:当前速度 pos_nxt:下一步位置
        p_dot_2_nxt = 2 * ((pos_nxt-pos_cur) - vel_cur * self.trajectory_info.delta_t) / \
                      self.trajectory_info.delta_t ** 2 # 下一步加速度
        # limit
        p_dot_2_nxt[0] = min(p_dot_2_nxt[0],self.UAV_info.limit[0])
        p_dot_2_nxt[1] = min(p_dot_2_nxt[1],self.UAV_info.limit[1])
        # 将加速度限制在 self.UAV_info.limit 中指定的最大值

        s = vel_cur * self.trajectory_info.delta_t + 0.5 * p_dot_2_nxt * self.trajectory_info.delta_t ** 2
        pos_nxt_real = pos_cur + s
        p_dot_nxt = vel_cur + p_dot_2_nxt * self.trajectory_info.delta_t
        # 根据当前速度和计算出的加速度来计算实际的下一个位置

        # 之前训练模型的时候没加G,测试之前模型的时候需要调成错的
        # F_nxt = self.UAV_info.uav_weight * (p_dot_2_nxt + self.g) # 对的
        F_nxt = p_dot_2_nxt * self.UAV_info.uav_weight
        # 根据无人机的重量和计算出的加速度来计算作用在无人机上的力。
        attitude_nxt = self.get_attitude_by_F(F_nxt)
        # 根据作用在无人机上的力来计算无人机的姿态
        return np.concatenate([pos_nxt_real,p_dot_nxt,p_dot_2_nxt,attitude_nxt],axis=0)

    def reset_clock(self):
        self.clock = self.trajectory_info.start_time
        return