import torch
import numpy as np

import DeepLearning_Module.util as util

# 接收输入数据 `X`、目标数据 `targets`、批处理大小 `batch_size` 和 `ModelUser`，并返回损失列表、TP/TN/FP/FN
# 列表以及所有正确预测的列表。具体步骤：
# 1. 将模型设置为评估模式。
# 2. 计算输入数据的形状参数。
# 3. 将输入数据和目标数据展平。
# 4. 计算批处理数量。
# 5. 初始化损失值和样本计数。
# 6. 遍历每个批处理，进行前向传播，计算损失，并收集预测结果。
# 7. 计算并打印损失日志。
# 8. 将输出转换为张量格式。
# 9. 计算并打印 TP/TN/FP/FN 列表。
# 10. 计算并打印所有正确预测的列表。
# 11. 重塑输出数据。
# 12. 返回输出、损失列表、TP/TN/FP/FN 列表和所有正确预测的列表。
def verify(X,targets,batch_size,ModelUser,show_progress=True): #
    if show_progress:
        print("Verify:")
    ModelUser.model_to_eval()

    attack_type_num = X.shape[0]
    attack_aim_type_num = X.shape[1]
    trajectory_start_point_num = X.shape[2]
    time_step_len = X.shape[3]
    uav_num = X.shape[5]
    output = [[],[],[],[]]

    flatten_SW = util.flatten_SW(X)
    flatten_target_label1, flatten_target_label2, \
        flatten_target_label3, flatten_target_compensation = \
        util.flatten_targets(targets[0], targets[1], targets[2], targets[3])

    sample_num = flatten_SW.shape[0]
    batch_num = sample_num // batch_size
    if batch_num * batch_size < sample_num:
        batch_num += 1
    sample_indexes = [i for i in range(sample_num)]

    loss_value = [0,0,0,0]
    sample_cnt = [0,0,0,0]
    for b_i in range(batch_num):
        X_batch,target_label1_batch,target_label2_batch,\
            target_label3_batch,target_compensation_batch = \
            util.get_batch_X_Target(
                flatten_SW,flatten_target_label1, flatten_target_label2,
                flatten_target_label3, flatten_target_compensation,
                sample_indexes[b_i * batch_size:(b_i + 1) * batch_size],
                ModelUser.device
            )
        if show_progress:
            print("\tbatch:[{}/{}] sample_num:[{}]".format(b_i + 1, batch_num, X_batch.shape[0]))

        ModelUser.forward(X_batch)
        loss_value_batch,sample_cnt_batch = ModelUser.calc_loss(
            target_label1_batch, target_label2_batch,
            target_label3_batch, target_compensation_batch
        )
        util.batch_out_GPU(
            X_batch,
            target_label1_batch,target_label2_batch,
            target_label3_batch,target_compensation_batch
        )

        pre_label1,pre_label2,pre_label3,pre_compensation = util.predict2array(
            ModelUser.pre_label1,ModelUser.pre_label2,
            ModelUser.pre_label3,ModelUser.pre_compensation
        )
        util.collect_output(
            output,ModelUser.alpha,
            pre_label1,pre_label2,
            pre_label3,pre_compensation
        )
        for i in range(len(loss_value)):
            loss_value[i] += loss_value_batch[i]
            sample_cnt[i] += sample_cnt_batch[i]

    loss_lst = get_loss_lst(loss_value,sample_cnt)
    print(get_loss_log(loss_lst))

    util.output2tensor(output)
    flatten_target_label1,flatten_target_label2,flatten_target_label3 = \
        util.label_targets2int_tensor(flatten_target_label1,flatten_target_label2,flatten_target_label3)

    TP_TN_FP_FN_lst = get_TP_lst(
        output,
        (flatten_target_label1,flatten_target_label2,flatten_target_label3)
    )
    print(get_TP_log(TP_TN_FP_FN_lst))

    all_right_lst = get_all_right_lst(
        output[0],output[1],output[2],
        flatten_target_label1,flatten_target_label2,flatten_target_label3
    )
    print(get_all_right_log(all_right_lst))

    util.reshape_output(
        output,
        attack_type_num,attack_aim_type_num,
        trajectory_start_point_num,time_step_len,uav_num
    )
    return output,loss_lst,TP_TN_FP_FN_lst,all_right_lst

def get_loss_lst(loss_value,sample_cnt):
    loss_lst = [
        loss_value[0] / sample_cnt[0] if sample_cnt[0] != 0 else -1,
        loss_value[1] / sample_cnt[1] if sample_cnt[1] != 0 else -1,
        loss_value[2] / sample_cnt[2] if sample_cnt[2] != 0 else -1,
        loss_value[3] / sample_cnt[3] if sample_cnt[3] != 0 else -1,
    ]
    return loss_lst

def get_loss_log(
        loss_lst
):
    log = "label1:[{:.6f}] label2:[{:.6f}] label3:[{:.6f}] compensation[{:.6f}]".format(
          loss_lst[0],loss_lst[1],loss_lst[2],loss_lst[3]
    )
    return log

def get_TP_TN_FP_FN(pre_label,tar_label):  # 计算TP/TN/FP/FN
    TP = torch.sum((pre_label == 1) & (tar_label == 1)).item()
    TN = torch.sum((pre_label == 0) & (tar_label == 0)).item()
    FP = torch.sum((pre_label == 1) & (tar_label == 0)).item()
    FN = torch.sum((pre_label == 0) & (tar_label == 1)).item()
    sum_ = TP + TN + FP + FN
    return {'TP':TP/sum_,'TN':TN/sum_,'FP':FP/sum_,'FN':FN/sum_}

def get_TP_lst(output,flatten_targets): # 计算TP/TN/FP/FN列表
    TP_TN_FP_FN_lst = []
    for i in range(len(flatten_targets)):
        TP_TN_FP_FN_lst.append(get_TP_TN_FP_FN(output[i], flatten_targets[i]))
    return TP_TN_FP_FN_lst

def get_TP_log(TP_TN_FP_FN_lst):
    log = "{:<9s}{:^10s}{:^10s}{:^10s}{:^10s}\n".format("Title","TP","FP","TN","FN")
    for i in range(len(TP_TN_FP_FN_lst)):
        log += "{:<9s}{:^10.6f}{:^10.6f}{:^10.6f}{:^10.6f}\n".format(
            'label'+str(i+1),TP_TN_FP_FN_lst[i]["TP"],TP_TN_FP_FN_lst[i]["FP"],
            TP_TN_FP_FN_lst[i]["TN"],TP_TN_FP_FN_lst[i]["FN"]
        )
    return log[:-1]

def get_all_right_lst(
        pre_label1,pre_label2,pre_label3,
        tar_label1,tar_label2,tar_label3
):
    sample_num = pre_label1.shape[0]
    rs_label2 = pre_label1 & pre_label2
    rs_label3 = pre_label1 & pre_label3
    all_right_1 = len(torch.where(torch.sum(tar_label1 == pre_label1, dim=1) == 5)[0])
    all_right_2 = len(torch.where(torch.sum(tar_label2 == rs_label2, dim=1) == 5)[0])
    all_right_3 = len(torch.where(torch.sum(tar_label3 == rs_label3, dim=1) == 5)[0])
    return all_right_1/sample_num,all_right_2/sample_num,all_right_3/sample_num

def get_all_right_log(all_right_lst):
    log = "All Right:\n"
    for i in range(len(all_right_lst)):
        log += "\tlabel{:s}:{:.6f}".format(str(i+1),all_right_lst[i])
    return log