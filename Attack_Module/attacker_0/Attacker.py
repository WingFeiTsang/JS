import numpy as np

from .Attacker_Uplink import Attacker_Uplink
from .Attacker_Downlink import Attacker_Downlink

class Attacker():
    def record_input(self,SW): # 记录输入

    def __init__(self,uav_weight,delte_t,AU_param_id=0,AD_param_id=0):
        self.A_U = Attacker_Uplink(uav_weight,delte_t,AU_param_id)
        self.A_D = Attacker_Downlink(AD_param_id)

    def update_attacker(self):
        self.A_U.update()
        self.A_D.update()

    def add_attack(
            self,
            SW,
            desired_state,real_state, # 期望位置、实际位置
            nxt_desired_position, # 下一步期望位置
            attack_types,attack_aims,start_point_num, # 攻击类型、攻击目标、起始点数
            fly_by_command_func,# 飞行指令函数
            compensated_indexes=None, # 补偿索引
    ): # 向滑动窗口中添加攻击
        targets_lst = [[[[[] for _ in range(start_point_num)] for _ in range(len(attack_aims))] for _ in range(len(attack_types))] for _ in range(4)]
        for at_i,at in enumerate(attack_types):
            for am_i,am in enumerate(attack_aims):
                for sp_i in range(start_point_num):
                    sample = SW[at_i,am_i,sp_i,-1]
                    if at == -1: # 无攻击
                        targets = self.attack_null(sample)
                    elif at == 0: # 下行攻击
                        targets = self.attack_0(sample,am)
                    elif at == 1: # 上行攻击
                        cur_real_state = SW[at_i,am_i,sp_i,-2]
                        cur_desired_state = desired_state[at_i][am_i][sp_i][-1]
                        targets = self.attack_1(
                            sample,am,
                            cur_real_state,cur_desired_state,
                            nxt_desired_position[at_i,am_i,sp_i],
                            fly_by_command_func,
                            compensated_indexes if compensated_indexes is None else compensated_indexes[at_i,am_i,sp_i],
                        )
                    elif at == 3: # 上下行攻击
                        cur_real_state = real_state[am_i][sp_i][-1]
                        cur_desired_state = desired_state[at_i][am_i][sp_i][-1]
                        targets,nxt_real_state = self.attack_3(
                            sample,am,
                            cur_real_state,cur_desired_state,
                            nxt_desired_position[at_i,am_i,sp_i],
                            fly_by_command_func,
                            compensated_indexes if compensated_indexes is None else compensated_indexes[0,am_i,sp_i],
                        )
                        real_state[am_i][sp_i].append(nxt_real_state)
                    else:
                        raise RuntimeError("No matching attack_type")
                    for i in range(len(targets)):
                        targets_lst[i][at_i][am_i][sp_i].append(targets[i])
        for i in range(4):
            targets_lst[i] = np.array(targets_lst[i])
        return targets_lst

    def get_target_label(self,uav_num,attack_type,aims=()): # 获取目标标签
        target_label1 = np.array([0]*uav_num) # 初始化目标标签
        target_label2 = np.array([0]*uav_num) # 下行攻击目标标签
        target_label3 = np.array([0]*uav_num) # 上行攻击目标标签
        for i in aims: # 遍历攻击目标
            target_label1[i] = 1
            if attack_type == 0:
                target_label2[i] = 1
            elif attack_type == 1:
                target_label3[i] = 1  # 上行是3
            elif attack_type == 3:
                target_label2[i] = 1
                target_label3[i] = 1
            else:
                pass
        return target_label1,target_label2,target_label3

    def attack_null(self,sample): # 无攻击
        uav_num = sample.shape[0]
        target_label1,target_label2,target_label3 = self.get_target_label(uav_num,-1) # 打上攻击目标标签
        target_compensation = sample[:,[-3,-2,-1]].copy()
        return target_label1, target_label2, target_label3, target_compensation

    def attack_0(self,sample,attack_aims): # 下行攻击
        uav_num = sample.shape[0]
        target_label1,target_label2,target_label3 = self.get_target_label(uav_num,0,attack_aims)
        target_compensation = sample[:,[-3,-2,-1]].copy() # 获取补偿

        for attack_aim in attack_aims:
            sample[attack_aim,[i for i in range(1,13)]] = self.A_D.execute(sample[attack_aim])
        return target_label1, target_label2, target_label3, target_compensation

    def attack_1(
            self,
            sample,attack_aims,
            cur_real_state,cur_desired_state,
            nxt_desired_position,
            fly_by_command_func,
            compensated_uav_id,
    ): # 上行攻击
        uav_num = sample.shape[0]
        target_label1,target_label2,target_label3 = self.get_target_label(uav_num,1,attack_aims)
        target_compensation = nxt_desired_position

        for attack_aim in attack_aims:
            target_compensation[attack_aim] = self.A_U.get_re_attacked_position(target_compensation[attack_aim])
            if not compensated_uav_id is None and compensated_uav_id[attack_aim] == 1:
                nxt_position = sample[attack_aim, [-3, -2, -1]].copy()
                sample[attack_aim, [i for i in range(1, 13)]] = self.A_U.execute_after_compensation(
                    cur_real_state[attack_aim], nxt_position, fly_by_command_func)
            else:
                sample[attack_aim,[i for i in range(1,13)]] = self.A_U.execute(
                    cur_real_state[attack_aim],
                    cur_desired_state[attack_aim],
                    sample[attack_aim]
                )

        return target_label1, target_label2, target_label3, target_compensation

    def attack_3( # 上下行攻击
            self,
            sample,attack_aims,
            cur_real_state,cur_desired_state,
            nxt_desired_position,
            fly_by_command_func,
            compensated_uav_id,
    ): # 形参： sample:样本,attack_aims:攻击目标,cur_real_state:当前实际状态,cur_desired_state:当前期望状态,nxt_desired_position:下一步期望位置,fly_by_command_func:飞行指令函数,compensated_uav_id:补偿无人机ID
        uav_num = sample.shape[0]
        target_label1,target_label2,target_label3 = self.get_target_label(uav_num,3,attack_aims)
        target_compensation = nxt_desired_position

        for attack_aim in attack_aims:
            target_compensation[attack_aim] = self.A_U.get_re_attacked_position(target_compensation[attack_aim])
            if not compensated_uav_id is None and compensated_uav_id[attack_aim] == 1:
                nxt_position = sample[attack_aim, [-3, -2, -1]].copy()
                sample[attack_aim, [i for i in range(1, 13)]] = self.A_U.execute_after_compensation(
                    cur_real_state[attack_aim], nxt_position, fly_by_command_func)
            else:
                sample[attack_aim,[i for i in range(1,13)]] = self.A_U.execute(
                    cur_real_state[attack_aim],
                    cur_desired_state[attack_aim],
                    sample[attack_aim]
                )
        nxt_real_state = sample.copy()
        for attack_aim in attack_aims:
            sample[attack_aim,[i for i in range(1,13)]] = self.A_D.execute(sample[attack_aim])
        return (target_label1, target_label2, target_label3, target_compensation), nxt_real_state