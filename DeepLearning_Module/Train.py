import random as rd
import DeepLearning_Module.util as util

def train(X, targets, batch_size, epoch, ModelUser, show_progress=True): # 训练
    if show_progress:
        print("Train:")
    ModelUser.model_to_train()  # 将模型设置为训练模式

    # 将输入数据和目标数据展平
    flatten_SW = util.flatten_SW(X)
    flatten_target_label1, flatten_target_label2, \
        flatten_target_label3, flatten_target_compensation = \
        util.flatten_targets(targets[0], targets[1], targets[2], targets[3])

    sample_num = flatten_SW.shape[0]  # 样本数量
    batch_num = sample_num // batch_size  # 批次数量
    if batch_num * batch_size < sample_num:
        batch_num += 1  # 如果有剩余样本，增加一个批次
    sample_indexes = [i for i in range(sample_num)]  # 样本索引列表
    rd.shuffle(sample_indexes)  # 随机打乱样本索引

    for e_i in range(epoch):  # 迭代每个epoch
        loss_value = [0, 0, 0, 0]  # 初始化损失值
        sample_cnt = [0, 0, 0, 0]  # 初始化样本计数
        for b_i in range(batch_num):  # 迭代每个批次
            # 获取当前批次的输入数据和目标数据
            X_batch, target_label1_batch, target_label2_batch, \
                target_label3_batch, target_compensation_batch = \
                util.get_batch_X_Target(
                    flatten_SW, flatten_target_label1, flatten_target_label2,
                    flatten_target_label3, flatten_target_compensation,
                    sample_indexes[b_i * batch_size:(b_i + 1) * batch_size],
                    ModelUser.device
                )
            if show_progress:
                print(get_batch_log(e_i + 1, epoch, b_i + 1, batch_num, X_batch.shape[0]))
            ModelUser.forward(X_batch)  # 前向传播
            # 计算损失值和样本计数
            loss_value_batch, sample_cnt_batch = ModelUser.calc_loss(
                target_label1_batch, target_label2_batch,
                target_label3_batch, target_compensation_batch
            )
            ModelUser.backward()  # 反向传播
            # 将当前批次的数据移出GPU
            util.batch_out_GPU(
                X_batch,
                target_label1_batch, target_label2_batch,
                target_label3_batch, target_compensation_batch
            )
            # 累加损失值和样本计数
            for i in range(len(loss_value)):
                loss_value[i] += loss_value_batch[i]
                sample_cnt[i] += sample_cnt_batch[i]
        print(get_epoch_log(e_i + 1, epoch, loss_value, sample_cnt))  # 打印当前epoch的日志
    return


def get_batch_log(cur_epoch,all_epoch,cur_batch,all_batch,sample_num):
    log = "\tepoch:[{:d}/{:d}] batch:[{:d}/{:d}] sample_num:[{:d}]".format(
        cur_epoch, all_epoch, cur_batch, all_batch, sample_num
    )
    return log

def get_epoch_log(
        cur_epoch,all_epoch,
        loss_value,sample_cnt
):
    log = "epoch:[{:d}/{:d}] label1:[{:.6f}] label2:[{:.6f}] label3:[{:.6f}] compensation[{:.6f}]".format(
          cur_epoch, all_epoch,
          loss_value[0] / sample_cnt[0],
          loss_value[1] / sample_cnt[1],
          loss_value[2] / sample_cnt[2],
          loss_value[3] / sample_cnt[3])
    return log