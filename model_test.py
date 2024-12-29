import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import matplotlib
matplotlib.use("TKAGG")
from matplotlib import pyplot as plt

from Trajectory_Module.IAA_calculator import get_area as calc_IAA
from DeepLearning_Module.Verify import verify

class ResultCollector():
    def __init__(self,timestamp):
        self.loss_collector = {}
        self.TP_collector = {}
        self.Acc_collector = {}
        self.all_right_collector = {}
        self.init_all_collectors()
        self.timestamp = timestamp

    def init_all_collectors(self):
        self.init_loss_collector()
        self.init_TP_collector()
        self.init_Acc_collector()
        self.init_ar_collector()
        return

    def add_one_step_record(self,round_,step,loss_lst,TP_lst,ar_lst):
        if round_ == -1:
            cur_index = "step[{:d}]".format(step)
        else:
            cur_index = "round[{:d}]_step[{:d}]".format(round_,step)

        self.loss_collector["Index"].append(cur_index)
        self.add_loss_collector(loss_lst)

        self.TP_collector["Index"].append(cur_index)
        self.add_TP_collector(TP_lst)

        self.Acc_collector["Index"].append(cur_index)
        self.add_Acc_collector(TP_lst)

        self.all_right_collector["Index"].append(cur_index)
        self.add_ar_collector(ar_lst)
        return

    def save_collector(self,save_root):
        save_root_obj = Path(save_root)
        self.save_loss_collector(self.timestamp,save_root_obj)
        self.save_TP_collector(self.timestamp,save_root_obj)
        self.save_Acc_collector(self.timestamp,save_root_obj)
        self.save_ar_collector(self.timestamp,save_root_obj)
        return

    def init_loss_collector(self):
        self.loss_collector = {
            "Index":[],
            "loss_label1":[],"loss_label2":[],
            "loss_label3":[],"loss_compensation":[],
        }
        return

    def init_TP_collector(self):
        self.TP_collector["Index"] = []
        tags = ("TP","FP","TN","FN")
        for i in range(3):
            for tag in tags:
                self.TP_collector["label{:d}_{:s}".format(i+1,tag)] = []
        return

    def init_Acc_collector(self):
        self.Acc_collector["Index"] = []
        tags = ("Acc","Precision","Recall","F1-Score")
        for i in range(3):
            for tag in tags:
                self.Acc_collector["label{:d}_{:s}".format(i+1,tag)] = []
        return

    def init_ar_collector(self):
        self.all_right_collector = {
            "Index": [],
            "all_right_label1":[],"all_right_label2":[],"all_right_label3":[]
        }
        return

    def add_loss_collector(self,loss_lst):
        self.loss_collector["loss_label1"].append(loss_lst[0])
        self.loss_collector["loss_label2"].append(loss_lst[1])
        self.loss_collector["loss_label3"].append(loss_lst[2])
        self.loss_collector["loss_compensation"].append(loss_lst[3])
        return

    def add_TP_collector(self,TP_lst):
        tags = ("TP", "FP", "TN", "FN")
        for i in range(3):
            for tag in tags:
                self.TP_collector["label{:d}_{:s}".format(i+1,tag)].append(TP_lst[i][tag])
        return


    def add_Acc_collector(self, TP_lst):
        # 计算准确率
        def calc_acc(TP_dic):
            acc = TP_dic["TP"] + TP_dic["TN"]
            return acc

        # 计算精确率
        def calc_precision(TP_dic):
            precision = TP_dic["TP"] / (TP_dic["TP"] + TP_dic["FP"]) if TP_dic["TP"] + TP_dic["FP"] != 0 else 0
            return precision

        # 计算召回率
        def calc_recall(TP_dic):
            recall = TP_dic["TP"] / (TP_dic["TP"] + TP_dic["FN"]) if TP_dic["TP"] + TP_dic["FN"] != 0 else 0
            return recall

        # 计算F1分数
        def calc_f1(TP_dic):
            precision = calc_precision(TP_dic)
            recall = calc_recall(TP_dic)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            return f1

        # 准确率指标的标签和对应的计算函数
        tags = ("Acc", "Precision", "Recall", "F1-Score")
        funcs = (calc_acc, calc_precision, calc_recall, calc_f1)

        # 遍历每个标签并计算指标
        for i in range(3):
            for t_i, tag in enumerate(tags):
                self.Acc_collector["label{:d}_{:s}".format(i + 1, tag)].append(funcs[t_i](TP_lst[i]))
        return

    def add_ar_collector(self,ar_lst): # 添加所有正确的记录
        self.all_right_collector["all_right_label1"].append(ar_lst[0])
        self.all_right_collector["all_right_label2"].append(ar_lst[1])
        self.all_right_collector["all_right_label3"].append(ar_lst[2])
        return

    def save_loss_collector(self,timestamp,save_root_obj): # 保存损失记录
        loss_save_name = "{}_loss_log.csv".format(timestamp)
        loss_df = pd.DataFrame(self.loss_collector)
        loss_df.to_csv(str(save_root_obj / loss_save_name))
        return

    def save_TP_collector(self,timestamp,save_root_obj): # 保存TP记录
        TP_save_name = "{}_TP_log.csv".format(timestamp)
        TP_df = pd.DataFrame(self.loss_collector)
        TP_df.to_csv(str(save_root_obj / TP_save_name))
        return

    def save_Acc_collector(self,timestamp,save_root_obj):
        Acc_save_name = "{}_Acc_log.csv".format(timestamp)
        Acc_df = pd.DataFrame(self.Acc_collector)
        Acc_df.to_csv(str(save_root_obj / Acc_save_name))
        return

    def save_ar_collector(self,timestamp,save_root_obj):
        ar_save_name = "{}_all_right_log.csv".format(timestamp)
        ar_df = pd.DataFrame(self.all_right_collector)
        ar_df.to_csv(str(save_root_obj / ar_save_name))
        return

class TrajectoryCollector():
    def __init__(self, X_lst, start_point_num, realTrajectories, TrajectoryGenerator):
        # 获取补偿后的轨迹
        self.trajectories = self.get_compensated_trajectories(X_lst)
        # 获取期望的轨迹
        self.desiredTrajectories = self.get_desired_trajectories(TrajectoryGenerator, start_point_num)
        # 真实轨迹
        self.realTrajectories = realTrajectories

    def get_compensated_trajectories(self, X_lst):
        # 获取起始时间步
        start_time_step = X_lst[:, :, :, 0, [0, 1]]
        # 获取轨迹
        trajectories = X_lst[:, :, :, :, -1]
        # 将起始时间步和轨迹拼接
        trajectories = np.concatenate([start_time_step, trajectories], axis=3)
        return trajectories

    def get_desired_trajectories(self, TrajectoryGenerator, start_point_num):
        # 获取时间步数
        time_step = self.trajectories.shape[3]
        # 计算时间间隔
        time_interval = TrajectoryGenerator.trajectory_info.period / start_point_num
        delta_t = TrajectoryGenerator.trajectory_info.delta_t
        desired_trajectories = []
        for sp_i in range(start_point_num):
            t = TrajectoryGenerator.clock + sp_i * time_interval
            desired_trajectories.append([])
            for ts in range(time_step):
                # 获取期望的群体状态
                desired_trajectories[-1].append(TrajectoryGenerator.get_swarm_state_by_t(t + ts * delta_t))
        return np.array(desired_trajectories)

    # aim_trajectory_id = ((attack_aim_id, start_point_id))
    def get_real_trajectory_IAA(self, uav_i, *aim_trajectory_id):
        print(self.realTrajectories.shape)
        # 获取真实轨迹
        aim_trajectory = self.realTrajectories[aim_trajectory_id[0][0], aim_trajectory_id[0][1]][:, uav_i, [1, 2]]
        # 获取期望轨迹
        desired_trajectory = self.desiredTrajectories[aim_trajectory_id[0][-1]][:, uav_i, [1, 2]]
        # 计算IAA
        IAA = calc_IAA(aim_trajectory, desired_trajectory)
        return IAA

    # aim_trajectory_id = ((attack_type_id,attack_aim_id,start_point_id))
    def get_attacked_trajectory_IAA(self,uav_i,*aim_trajectory_id):
        aim_trajectory = self.trajectories[aim_trajectory_id[0][0],aim_trajectory_id[0][1],aim_trajectory_id[0][2]][:,uav_i,[1, 2]]
        desired_trajectory = self.desiredTrajectories[aim_trajectory_id[0][-1]][:,uav_i,[1,2]]
        IAA = calc_IAA(aim_trajectory,desired_trajectory)
        return IAA

    def init_figure(self,figsize=(3,3),dpi=300):
        plt.figure(figsize=figsize,dpi=dpi)
        return

    # 在3类攻击下画实际路径和预期路径
    # aim_trajectory_id = ((attack_aim_id, start_point_id))
    def plt_real_trajectory(self, uav_i, *aim_trajectory_id, start=0, end=200, plt_aims=(1, 2)):
        self.init_figure()
        # 获取真实轨迹的x和y坐标
        x = self.realTrajectories[aim_trajectory_id[0][0], aim_trajectory_id[0][1]][start:end, uav_i, plt_aims[0]]
        y = self.realTrajectories[aim_trajectory_id[0][0], aim_trajectory_id[0][1]][start:end, uav_i, plt_aims[1]]
        plt.plot(x, y, label="real")
        # 获取期望轨迹的x和y坐标
        x = self.desiredTrajectories[aim_trajectory_id[0][-1]][start:end, uav_i, plt_aims[0]]
        y = self.desiredTrajectories[aim_trajectory_id[0][-1]][start:end, uav_i, plt_aims[1]]
        plt.plot(x, y, label="normal")
        plt.legend()
        plt.show()
        return

    # 在0和1类攻击下画补偿路径和预期路径
    # aim_trajectory_id = ((attack_type_id,attack_aim_id,start_point_id))
    def plt_trajectory(self,uav_i,*aim_trajectory_id,start=0,end=200,plt_aims=(1,2)):
        self.init_figure()
        x = self.trajectories[aim_trajectory_id[0][0],aim_trajectory_id[0][1],aim_trajectory_id[0][2]][start:end,uav_i,plt_aims[0]]
        y = self.trajectories[aim_trajectory_id[0][0],aim_trajectory_id[0][1],aim_trajectory_id[0][2]][start:end,uav_i,plt_aims[1]]
        save_path = Path("trajectory_data_u25_1.csv")
        data = {
            "attacked_x": x,
            "attacked_y": y,

        }
        df = pd.DataFrame(data)
        df.to_csv(save_path)
        plt.plot(x,y, label="attacked")
        x = self.desiredTrajectories[aim_trajectory_id[0][-1]][start:end,uav_i, plt_aims[0]]
        y = self.desiredTrajectories[aim_trajectory_id[0][-1]][start:end,uav_i, plt_aims[1]]
        plt.plot(x,y, label="normal")
        plt.legend()
        plt.show()
        return

    def save_trajectory(self,save_root,*aim_trajectory_id):
        save_root_obj = Path(save_root)
        timestamp = str(datetime.datetime.now())[:-7].replace(":", "-")
        save_path = str(save_root_obj/timestamp)+".csv"
        aim_trajectory = self.trajectories[aim_trajectory_id]
        dic = {}
        for uav_i in range(aim_trajectory.shape[1]):
            for t_i,tag in enumerate(("x","y")):
                dic["UAV_{:d}_{:s}".format(uav_i,tag)] = aim_trajectory[:,uav_i,t_i+1]
        df = pd.DataFrame(dic)
        df.to_csv(save_path)
        return

def verify_one_step(X_one_step,targets_one_step,batch_size,ModelUser):  # 验证
    X_one_step = np.expand_dims(X_one_step,axis=3)
    for i in range(len(targets_one_step)):
        targets_one_step[i] = np.expand_dims(targets_one_step[i],axis=3)
    _,loss_lst,TP_TN_FP_FN_lst,all_right_lst = verify(X_one_step,targets_one_step,batch_size,ModelUser)
    return loss_lst,TP_TN_FP_FN_lst,all_right_lst