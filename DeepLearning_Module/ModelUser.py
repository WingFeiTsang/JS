import torch
import torch.optim as opt
import datetime
import yaml
from pathlib import Path

from .Model import FeatureMining,MultiTask,Vgg16,TransformerEncoder as Transformer

class ModelUser():
    def __init__(self,configure_path,timestamp,uav_num):
        self.load_config(configure_path,uav_num)
        self.timestamp = timestamp

    def load_config(self,configure_path,uav_num):
        f = open(configure_path,"r",encoding="UTF-8")
        configure = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        self.device = "cuda:0" if configure["use_GPU"] and torch.cuda.is_available() else "cpu" # 使用GPU
        self.alpha = configure["alpha"]

        self.FM_config = configure["feature_mining"] # 特征提取模型配置
        self.MT_config = configure["multi_task"] # 多任务模型配置
        self.loss_F1_name = configure["loss_function1"] # 损失函数1
        if self.loss_F1_name == "CE": # 交叉熵损失
            self.loss_F1 = torch.nn.CrossEntropyLoss()
        elif self.loss_F1_name == "BCE": # 二元交叉熵损失
            self.loss_F1 = torch.nn.BCELoss()
        else:
            raise RuntimeError("No matching loss_function1")
        self.loss_F2_name = configure["loss_function2"]
        if self.loss_F2_name == "MSE": # 均方误差损失
            self.loss_F2 = torch.nn.MSELoss()
        else:
            raise RuntimeError("No matching loss_function2")
        self.init_model(uav_num)

    def init_model(self,uav_num): # 初始化模型
        if self.FM_config["type"] == "cnn_lstm": # 选择LSTM模型
            self.FM_model = FeatureMining(
                hidden_size=self.FM_config["hidden_size"],
                layer=self.FM_config["layer"],
                dropout=self.FM_config["dropout"],
            ) # 特征提取模型
            self.FM_model.to(self.device)
            self.FM_model_opt = opt.Adam(filter(lambda p: p.requires_grad, self.FM_model.parameters()),
                                     lr=self.FM_config["lr"]) # 优化器
        elif self.FM_config["type"] == "vgg16": # VGG16模型
            self.FM_model = Vgg16()
        elif self.FM_config["type"] == "transformer": # Transformer模型
            self.FM_model = Transformer()
        else:
            raise RuntimeError("No matching FeatureMining type")

        if self.FM_config["load_switch"] and len(self.FM_config["load_path"]) > 0: # 加载模型
            # new_state_dict = torch.load(self.FM_config["load_path"])
            # new_state_dict = torch.load(self.FM_config["load_path"])
            # self.FM_model.load_state_dict(new_state_dict)
            new_model = torch.load(self.FM_config["load_path"])
            new_state_dict = new_model.state_dict()
            self.FM_model.load_state_dict(new_state_dict)
        self.FM_model.eval()

        self.MT_model = MultiTask( # 多任务模型
            input_channel=self.MT_config["input_channel"],
            uav_num=uav_num,
            dropout=self.MT_config["dropout"],
        )
        self.MT_model.to(self.device)
        self.MT_model_opt = opt.Adam(filter(lambda p: p.requires_grad, self.MT_model.parameters()),
                                   lr=self.MT_config["lr"])
        if self.MT_config["load_switch"] and len(self.MT_config["load_path"]) > 0:
            # new_state_dict = torch.load(self.MT_config["load_path"])
            # self.MT_model.load_state_dict(new_state_dict)
            new_model = torch.load(self.MT_config["load_path"])
            new_state_dict = new_model.state_dict()
            self.MT_model.load_state_dict(new_state_dict)
        self.MT_model.eval() # 模型评估
        return

    def model_to_train(self):
        self.FM_model.train()
        self.MT_model.train()

    def model_to_eval(self):
        self.FM_model.eval()
        self.MT_model.eval()

    def forward(self,X):
        self.FM_model_opt.zero_grad()
        self.MT_model_opt.zero_grad()

        if self.FM_config["type"] == "cnn_lstm":
            lstm_output, (h_n, c_n) = self.FM_model(X)
            multi_label_X = torch.concat([h_n[-1].unsqueeze(1), c_n[-1].unsqueeze(1)], dim=1)
        else:
            multi_label_X = self.FM_model(X)

        self.pre_label1, self.pre_label2, self.pre_label3, self.pre_compensation = self.MT_model(multi_label_X)
        return

    def calc_loss(self,tar_label1,tar_label2,tar_label3,tar_compensation): # 计算损失
        def get_attack_indexes(target_label1, perdict_label1): # 获取攻击索引
            indexes = torch.where(((target_label1==1) | (perdict_label1>=self.alpha)) == True)
            return indexes

        def get_downlink_attack_indexes(target_label1, target_label3, perdict_label1, perdict_label3): # 获取下行攻击索引
            indexes = torch.where((((target_label1==1) | (perdict_label1>=self.alpha)) &
                                  ((target_label3==1) | (perdict_label3>=self.alpha))) == True)
            return indexes

        loss_value = [0,0,0,0]
        sample_cnt = [0,0,0,0]
        if self.loss_F1_name == "BCE": # 二元交叉熵损失
            self.loss_label1 = self.loss_F1(
                self.pre_label1.flatten(),
                tar_label1.flatten()
            ) # 计算损失
            loss_value[0] += self.loss_label1.item() * self.pre_label1.flatten().shape[0]
            sample_cnt[0] += self.pre_label1.flatten().shape[0]
            attack_indexes = get_attack_indexes(tar_label1, self.pre_label1)
            if attack_indexes[0].shape[0] > 0: # 如果攻击索引大于0
                self.loss_label2 = self.loss_F1(
                    self.pre_label2[attack_indexes[0],attack_indexes[1]],
                    tar_label2[attack_indexes[0],attack_indexes[1]],
                )
                self.loss_label3 = self.loss_F1(
                    self.pre_label3[attack_indexes[0],attack_indexes[1]],
                    tar_label3[attack_indexes[0],attack_indexes[1]],
                )
                loss_value[1] += self.loss_label2.item() * attack_indexes[0].shape[0]
                loss_value[2] += self.loss_label3.item() * attack_indexes[0].shape[0]
                sample_cnt[1] += attack_indexes[0].shape[0]
                sample_cnt[2] += attack_indexes[0].shape[0]
            else:
                self.loss_label2 = None
                self.loss_label3 = None

        elif self.loss_F1_name == "CE": # 交叉熵损失
            self.loss_label1 = self.loss_F1(
                self.pre_label1,
                tar_label1
            )
            loss_value[0] += self.loss_label1.item() * self.pre_label1.shape[0] # 计算损失
            sample_cnt[0] += self.pre_label1.shape[0] # 样本数量
            attack_indexes = get_attack_indexes(tar_label1, self.pre_label1) # 获取攻击索引
            if attack_indexes[0].shape[0] > 0: # 如果攻击索引大于0
                self.loss_label2 = self.loss_F1(
                    self.pre_label2[attack_indexes[0]],
                    tar_label2[attack_indexes[0]],
                )
                self.loss_label3 = self.loss_F1(
                    self.pre_label3[attack_indexes[0]],
                    tar_label3[attack_indexes[0]],
                )
                # 根据攻击索引计算其他损失值
                loss_value[1] += self.loss_label2.item() * attack_indexes[0].shape[0]
                loss_value[2] += self.loss_label3.item() * attack_indexes[0].shape[0]
                sample_cnt[1] += attack_indexes[0].shape[0]
                sample_cnt[2] += attack_indexes[0].shape[0]

            else:
                self.loss_label2 = None
                self.loss_label3 = None
        else:
            raise RuntimeError("No matching loss_function1")

        downlink_attack_indexes = get_downlink_attack_indexes(tar_label1, tar_label3,
                                                              self.pre_label1, self.pre_label3)
        if downlink_attack_indexes[0].shape[0] > 0:
            self.loss_compensation = self.loss_F2(
                self.pre_compensation[downlink_attack_indexes[0], downlink_attack_indexes[1]],
                tar_compensation[downlink_attack_indexes[0], downlink_attack_indexes[1]]
            )
            loss_value[3] += self.loss_compensation.item() * downlink_attack_indexes[0].shape[0]
            sample_cnt[3] += downlink_attack_indexes[0].shape[0]
        else:
            self.loss_compensation = None
        return loss_value, sample_cnt

    def backward(self):
        self.loss_label1.backward(retain_graph=True) # 反向传播
        if not self.loss_label2 is None:
            self.loss_label2.backward(retain_graph=True)
        if not self.loss_label3 is None:
            self.loss_label3.backward(retain_graph=True)
        if not self.loss_compensation is None:
            self.loss_compensation.backward(retain_graph=True)
        self.FM_model_opt.step()
        self.MT_model_opt.step()

    def save_model(self,save_root,cur_round=None): # 保存模型
        save_root_obj = Path(save_root)
        self.save_FM(save_root_obj,cur_round)
        self.save_MT(save_root_obj,cur_round)
        return

    def save_FM(self,save_root_obj,cur_round): # 保存特征提取模型
        if cur_round is None:
            FM_save_name = "{}_FM.p".format(self.timestamp)
        else:
            FM_save_name = "{}_FM_round[{:d}].p".format(self.timestamp,cur_round)
        FM_save_path = str(save_root_obj/FM_save_name)
        torch.save(self.FM_model,FM_save_path)
        return

    def save_MT(self,save_root_obj,cur_round): # 保存多任务模型
        if cur_round is None:
            MT_save_name = "{}_MT.p".format(self.timestamp)
        else:
            MT_save_name = "{}_MT_round[{:d}].p".format(self.timestamp,cur_round)
        MT_save_path = str(save_root_obj/MT_save_name)
        torch.save(self.MT_model,MT_save_path)
        return
