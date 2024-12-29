import torch
import numpy as np

def flatten_SW(SW):
    time_step = SW.shape[-3]
    uav_n = SW.shape[-2]
    vector_len = SW.shape[-1]
    return SW.reshape((-1,time_step,uav_n,vector_len))[:,:,:,1:]

def flatten_targets(target_label1,target_label2,target_label3,target_compensation):
    uav_n = target_label1.shape[-1]
    vector_len = target_compensation.shape[-1]
    flatten_target_label1 = target_label1.reshape((-1, uav_n))
    flatten_target_label2 = target_label2.reshape((-1, uav_n))
    flatten_target_label3 = target_label3.reshape((-1, uav_n))
    flatten_target_compensation = target_compensation.reshape((-1, uav_n, vector_len))[:,:,:-1]
    return flatten_target_label1,flatten_target_label2,flatten_target_label3,flatten_target_compensation

def array2tensor(arr):
    ten = torch.tensor(arr,dtype=torch.float32)
    return ten

def tensor2array(ten):
    return ten.detach().cpu().numpy()

def binarizate(out,alpha):
    rs = np.where(out>=alpha,1,0).astype(int)
    return rs

def out_GPU(tensor):
    tensor.to("cpu")

def get_batch_X_Target(
        flatten_SW,
        target_label1,target_label2,
        target_label3,target_compensation,
        sample_indexes,device
):
    X_batch = array2tensor(
        flatten_SW[sample_indexes]
    ).to(device)
    tar_label1_batch = array2tensor(
        target_label1[sample_indexes]
    ).to(device)
    tar_label2_batch = array2tensor(
        target_label2[sample_indexes]
    ).to(device)
    tar_label3_batch = array2tensor(
        target_label3[sample_indexes]
    ).to(device)
    tar_compensation_batch = array2tensor(
        target_compensation[sample_indexes]
    ).to(device)
    return X_batch,tar_label1_batch,tar_label2_batch,\
        tar_label3_batch,tar_compensation_batch

def batch_out_GPU(
        X_batch,
        tar_label1_batch,tar_label2_batch,
        tar_label3_batch,tar_compensation_batch
):
    out_GPU(X_batch)
    out_GPU(tar_label1_batch)
    out_GPU(tar_label2_batch)
    out_GPU(tar_label3_batch)
    out_GPU(tar_compensation_batch)
    return

def label_targets2int_tensor(flatten_target_label1,flatten_target_label2,flatten_target_label3):
    flatten_target_label1 = torch.tensor(flatten_target_label1,dtype=torch.int)
    flatten_target_label2 = torch.tensor(flatten_target_label2,dtype=torch.int)
    flatten_target_label3 = torch.tensor(flatten_target_label3,dtype=torch.int)
    return flatten_target_label1,flatten_target_label2,flatten_target_label3

def predict2array(pre_label1,pre_label2,pre_label3,pre_compensation):
    pre_label1.to("cpu")
    pre_label2.to("cpu")
    pre_label3.to("cpu")
    pre_compensation.to("cpu")
    pre_label1_array = pre_label1.detach().cpu().numpy()
    pre_label2_array = pre_label2.detach().cpu().numpy()
    pre_label3_array = pre_label3.detach().cpu().numpy()
    pre_compensation_array = pre_compensation.detach().cpu().numpy()
    return pre_label1_array,pre_label2_array,pre_label3_array,pre_compensation_array

def collect_output(output,alpha,pre_label1,pre_label2,pre_label3,pre_compensation):
    output[0].append(binarizate(pre_label1, alpha))
    output[1].append(binarizate(pre_label2, alpha))
    output[2].append(binarizate(pre_label3, alpha))
    output[3].append(pre_compensation)
    return

def output2array(output):
    output[0] = np.concatenate(output[0],axis=0)
    output[1] = np.concatenate(output[1],axis=0)
    output[2] = np.concatenate(output[2],axis=0)
    output[3] = np.concatenate(output[3],axis=0)
    return

def output2tensor(output):
    output[0] = torch.tensor(np.concatenate(output[0],axis=0))
    output[1] = torch.tensor(np.concatenate(output[1],axis=0))
    output[2] = torch.tensor(np.concatenate(output[2],axis=0))
    output[3] = torch.tensor(np.concatenate(output[3],axis=0))
    return

def reshape_output(output,*axis_tuple):
    axis_lst = list(axis_tuple)
    vector_len = output[3][-1].shape[-1]
    axis_lst.append(vector_len)

    output[0] = output[0].reshape(axis_tuple)
    output[1] = output[1].reshape(axis_tuple)
    output[2] = output[2].reshape(axis_tuple)
    output[3] = output[3].reshape(axis_lst)
    return