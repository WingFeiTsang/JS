import shutil
import numpy as np
from pathlib import Path

from Trajectory_Module.Generator import Generator as TrajectoryGenerator
from Attack_Module.Attacker import Attacker
from DataCollection_Module.DataCollector import DataCollector
from DeepLearning_Module.ModelUser import ModelUser
from DeepLearning_Module.Compensator import Compensator
from model_test import verify_one_step,ResultCollector

def init_module(
        timestamp,batch_size,
        TGenerator_config_path,DataCollection_config_path,ModelUser_config_path
):
    tGenerator = TrajectoryGenerator(TGenerator_config_path)
    attacker = Attacker(
        tGenerator.UAV_info.uav_weight,
        tGenerator.trajectory_info.delta_t
    ) # 初始化attacker
    dataCollector = DataCollector(
        DataCollection_config_path,
        tGenerator, attacker
    ) # 初始化dataCollector
    modelUser = ModelUser(ModelUser_config_path,timestamp,tGenerator.UAV_info.uav_num)
    compensator = Compensator(batch_size, modelUser)
    # 使用 ModelUser_config_path 配置文件路径、时间戳 timestamp 和无人机数量初始化 ModelUser
    resultCollector = ResultCollector(timestamp)
    return dataCollector,compensator,resultCollector

def step_loop(compensator,dataCollector,step,verify_by_step=False,resultCollector=None):
    X_lst = []
    targets_lst = [[],[],[],[]] # 用于存储每个步骤的数据和目标？
    dataCollector.Attacker.reset_param()
    targets = dataCollector.init_sliding_window()
    # 将当前滑动窗口 dataCollector.SW 的副本添加到 X_lst。
    # 将当前目标 targets 添加到 targets_lst。
    # 调用 compensator.execute(dataCollector.SW) 执行补偿操作。
    # 调用 dataCollector.update_sliding_window() 更新滑动窗口。
    # 调用 compensator.update_compensation_to_window 更新滑动窗口中的补偿信息。
    # 调用 dataCollector.add_attack_to_SW 更新目标。
    for s in range(step):
        X_lst.append(dataCollector.SW.copy())
        for i in range(len(targets_lst)):
            targets_lst[i].append(targets[i])
        compensator.execute(dataCollector.SW)
        dataCollector.update_sliding_window()
        compensator.update_compensation_to_window(
            dataCollector.SW,
            compensator.compensated_flag,
            compensator.pre_compensation

        )
        targets = dataCollector.add_attack_to_SW(compensator.compensated_flag)
        if verify_by_step and not resultCollector is None: # 如果 verify_by_step 为 True 并且 resultCollector 不为空
            print("\nStep:",s+1)
            loss_lst,TP_TN_FP_FN_lst,all_right_lst = verify_one_step(
                X_lst[-1],[targets_lst[i][-1] for i in range(len(targets_lst))],
                compensator.batch_size,compensator.ModelUser
            ) # 调用 verify_one_step 函数验证当前步骤的结果
            resultCollector.add_one_step_record(
                -1,s+1,loss_lst,TP_TN_FP_FN_lst,all_right_lst) # 将验证结果添加到 resultCollector 中
    X_lst = np.stack(X_lst,axis=3)
    for i in range(len(targets_lst)): # 将 targets_lst 中的每个元素转换为 np.ndarray 类型
        targets_lst[i] = np.stack(targets_lst[i],axis=3)
    return X_lst, targets_lst

def save_config_file(timestamp,save_root,config_path_lst): # 保存配置文件
    save_root_obj = Path(save_root)
    for cp in config_path_lst:
        cp_obj = Path(cp)
        dst_path = str(save_root_obj/"{:s}_{:s}".format(timestamp,cp_obj.name))
        shutil.copy(cp,dst_path)