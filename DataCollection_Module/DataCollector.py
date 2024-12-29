import yaml
import itertools as it
import numpy as np

class DataCollector():
    def __init__(self,configure_path,trajectory_generator,Attacker):
        self.TGenerator = trajectory_generator
        self.Attacker = Attacker
        self.load_config(configure_path)

    # 从配置文件中读取参数
    def load_config(self,config_path):
        f = open(config_path,"r",encoding="UTF-8")
        config = yaml.load(f,Loader=yaml.FullLoader) # 读取配置文件
        f.close()

        self.trajectory_start_point_num = config["tarjectory_start_point_num"] # 轨迹起始点数
        self.time_step = config["time_step"] # 滑动窗口大小
        self.time_interval = self.TGenerator.trajectory_info.period / self.trajectory_start_point_num # 滑动窗口的时间间隔
        self.attack_types = config["attack_types"] # 攻击类型
        self.attack_aim_num = config["attack_aim_num"] # 攻击目标数量
        self.attack_aim_type = config["attack_aim_type"] # 攻击目标类型
        self.attack_aim_lst = sorted(list(it.combinations( # 攻击目标列表
            list(range(self.TGenerator.UAV_info.uav_num)),
            self.attack_aim_num)
        ))[:self.attack_aim_type] # 从所有可能的攻击目标中选取指定数量的攻击目标
        if 3 in self.attack_types:
            self.real_state = []
            for i in range(self.attack_aim_type):
                self.real_state.append([])
                for j in range(self.trajectory_start_point_num):
                    self.real_state[i].append([])
        return

    # 计算并存储下一步期望位置
    def get_nxt_desired_position(self):
        nxt_desired_position = []
        for at_i in range(len(self.attack_types)):
            nxt_desired_position.append([])
            for am_i in range(self.attack_aim_type):
                nxt_desired_position[-1].append([])
                for sp_i in range(self.trajectory_start_point_num): # 从当前时刻开始，计算下一步期望位置
                    t = self.TGenerator.clock + self.TGenerator.trajectory_info.delta_t * sp_i
                    temp = self.TGenerator.get_swarm_state_by_t(t)[:,[1,2,3]]
                    nxt_desired_position[-1][-1].append(temp)
        self.nxt_desired_position = np.array(nxt_desired_position)
        return

    #  初始化期望位置和实际位置
    def init_desired_real_state(self):
        self.desired_state = []
        for at_i in range(len(self.attack_types)):
            self.desired_state.append([])
            for aa_i in range(self.attack_aim_type):
                self.desired_state[at_i].append([])
                for sp_i in range(self.trajectory_start_point_num):
                    self.desired_state[at_i][aa_i].append([self.SW[-1, aa_i, sp_i, 0].copy(), self.SW[-1, aa_i, sp_i, 1].copy()])
        # real_state只服务于3类攻击
        self.real_state = []
        for aa_i in range(self.attack_aim_type):
            self.real_state.append([])
            for sp_i in range(self.trajectory_start_point_num):
                self.real_state[aa_i].append([self.SW[-1, aa_i, sp_i, 0].copy(), self.SW[-1, aa_i, sp_i, 1].copy()])

    # 更新期望位置
    def update_desired_state(self):
        for at_i in range(len(self.attack_types)):
            for aa_i in range(self.attack_aim_type):
                for sp_i in range(self.trajectory_start_point_num):
                    t = self.desired_state[at_i][aa_i][sp_i][-1][0,0] + self.TGenerator.trajectory_info.delta_t
                    self.desired_state[at_i][aa_i][sp_i].append(self.TGenerator.get_swarm_state_by_t(t))

    # 初始化滑动窗口
    def init_sliding_window(self):
        self.reset()
        self.SW = []
        for sp_i in range(self.trajectory_start_point_num):
            t = self.TGenerator.clock + sp_i * self.time_interval # 计算当前时刻
            window = []
            for ts_i in range(self.time_step):
                window.append(self.TGenerator.get_swarm_state_by_t(t+ts_i*self.TGenerator.trajectory_info.delta_t))
            self.SW.append(window)
        self.TGenerator.clock += self.TGenerator.trajectory_info.delta_t * self.time_step # 更新时钟
        self.SW = self.arr_repeat(np.array(self.SW),(len(self.attack_types),self.attack_aim_type)) # 重复数组
        self.init_desired_real_state() # 初始化期望位置和实际位置
        targets = self.add_attack_to_SW()
        return targets

    # 更新滑动窗口
    def update_sliding_window(self):
        nxt_time_step = []
        for sp_i in range(self.trajectory_start_point_num):
            t = self.TGenerator.clock + sp_i * self.time_interval
            nxt_time_step.append(self.TGenerator.get_swarm_state_by_t(t)) # 获取下一步状态
        self.TGenerator.clock_update()
        nxt_time_step = self.arr_repeat(np.array(nxt_time_step),(len(self.attack_types),self.attack_aim_type)) # 重复数组
        self.SW[:, :, :, :-1] = self.SW[:, :, :, 1:] # 更新滑动窗口
        self.SW[:, :, :, -1] = nxt_time_step
        return

    # 向滑动窗口中添加攻击
    def add_attack_to_SW(self,compensated_indexes=None):
        self.get_nxt_desired_position()
        targets = self.Attacker.add_attack(
            self.SW,
            self.desired_state, self.real_state,
            self.nxt_desired_position,
            self.attack_types,self.attack_aim_lst,self.trajectory_start_point_num,
            self.TGenerator.get_single_state_by_command,
            compensated_indexes
        )
        self.Attacker.update_attacker() # 更新
        self.update_desired_state()
        return targets

    # 重复数组
    def arr_repeat(self,arr,repeat_tuple):
        out = arr
        for i in range(len(repeat_tuple)-1,-1,-1): # 从后往前遍历
            out = [out]
            for r in range(repeat_tuple[i]-1):
                out.append(out[-1].copy()) # 复制数组
            out = np.array(out)
        return np.array(out)

    def reset(self):
        self.TGenerator.reset_clock()
        return