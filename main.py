import yaml
import datetime
import numpy as np

from util import init_module,step_loop,save_config_file
from model_test import TrajectoryCollector
from DeepLearning_Module.Train import train
from DeepLearning_Module.Verify import verify

def iterative_train(
        timestamp,
        round,step,epoch,batch_size,
        model_save_path,log_save_path,
        TGenerator_config_path,
        DataCollection_config_path,
        ModelUser_config_path
): # 迭代训练
    dataCollector, compensator,resultCollector = init_module(
        timestamp,
        batch_size, TGenerator_config_path,
        DataCollection_config_path,ModelUser_config_path
    )

    for r in range(round):
        dataCollector.reset()
        for s in range(step):
            # print当前轮次和步骤信息。
            # 调用 step_loop 函数，执行补偿操作并返回数据 X 和目标 targets。
            # 调用 train 函数，使用数据 X 和目标 targets 进行训练。
            # 调用 verify 函数，验证当前步骤的结果，并返回损失列表 loss_lst、TP/TN/FP/FN 列表 TP_TN_FP_FN_lst 和所有正确的列表 all_right_lst。
            # 调用 resultCollector.add_one_step_record 函数，记录当前步骤的验证结果。
            print("\n{:s}Round:{:d} Step:{:d}{:s}".format("-"*20,r+1,s+1,"-"*20))
            X, targets = step_loop(compensator,dataCollector,s+1)
            train(X,targets,batch_size,epoch,compensator.ModelUser)
            _,loss_lst,TP_TN_FP_FN_lst,all_right_lst = \
                verify(X,targets,batch_size,compensator.ModelUser,show_progress=True)
            resultCollector.add_one_step_record(
                r+1,s+1,loss_lst,TP_TN_FP_FN_lst,all_right_lst)
            # TP_TN_FP_FN_lst，True Positives / True Negatives / False Positives / False Negatives
        if len(model_save_path) > 0:
            compensator.ModelUser.save_model(model_save_path,r)
    if len(log_save_path) > 0:
        resultCollector.save_collector(log_save_path)
        record_input = np.array(dataCollector.Attacker.record_input)
        record_type3_after_0 = np.array(dataCollector.Attacker.record_type3_after_0)
        record_type3_after_1 = np.array(dataCollector.Attacker.record_type3_after_1)
        #存储攻击者的输入和输出，文件名中添加时间戳
        np.save(log_save_path + "/" + timestamp + "_record_input"  + ".npy", record_input)
        np.save(log_save_path + "/" + timestamp +  "_record_type3_after_0" + ".npy", record_type3_after_0)
        np.save(log_save_path + "/" + timestamp +  "_record_type3_after_1" + ".npy", record_type3_after_1)
    return

def test_for_step(
        timestamp,
        step,batch_size,
        log_save_path,
        TGenerator_config_path,
        DataCollection_config_path,
        ModelUser_config_path
):
    dataCollector, compensator, resultCollector = init_module(
        timestamp,
        batch_size, TGenerator_config_path,
        DataCollection_config_path, ModelUser_config_path
    ) # 初始化 dataCollector、compensator 和 resultCollector

    X,_ = step_loop(
        compensator,dataCollector,step,
        verify_by_step=True,
        resultCollector=resultCollector,
    ) # 进行训练（滑动窗口），获取 X 和 targets_lst
    TCollector = TrajectoryCollector(
        X,dataCollector.trajectory_start_point_num,
        np.array(dataCollector.real_state),dataCollector.TGenerator
    ) # 初始化 TrajectoryCollector
    IAA_0 = TCollector.get_real_trajectory_IAA(0, [0, 0, 0])
    print("REAL_IAA_uav0:",IAA_0)
    IAA_1 = TCollector.get_real_trajectory_IAA(1, [0, 0, 0])
    print("REAL_IAA_uav1:",IAA_1)
    attacked_iaa_0 = TCollector.get_attacked_trajectory_IAA(0, [0,0,0])
    print("ATTACK_IAA_0:",attacked_iaa_0)
    attacked_iaa_1 = TCollector.get_attacked_trajectory_IAA(1, [0,0,0])
    print("ATTACK_IAA_1:",attacked_iaa_1)
    TCollector.plt_real_trajectory(0, [0, 0]) # 绘制真实轨迹
    TCollector.plt_trajectory(1, [0, 0, 0]) # 绘制轨迹
    if len(log_save_path) > 0:
        resultCollector.save_collector(log_save_path)
    return

def main(func):
    timestamp = str(datetime.datetime.now())[:-7].replace(":", "-")
    if func == "train":
        f = open("train_configure.yml", "r", encoding="UTF-8")
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        # 读取配置文件
        save_config_file(
            timestamp,config["config_save_root"],
            (
                "train_configure.yml",
                config["TGenerator_config_path"],
                config["DataCollection_config_path"],
                config["ModelUser_config_path"]
            )
        )
        # 保存配置文件
        iterative_train(
            timestamp,
            config["round"], config["step"], config["epoch"], config["batch_size"],
            config["model_save_root"],config["log_save_root"],
            config["TGenerator_config_path"],
            config["DataCollection_config_path"],
            config["ModelUser_config_path"]
        ) # 迭代训练
    elif func == "test":
        f = open("test_configure.yml", "r", encoding="UTF-8")
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        test_for_step(
            timestamp,
            config["step"],config["batch_size"],
            config["log_save_root"],
            config["TGenerator_config_path"],
            config["DataCollection_config_path"],
            config["ModelUser_config_path"]
        )
    return

if __name__ == '__main__':
    func = "test"
    main(func)