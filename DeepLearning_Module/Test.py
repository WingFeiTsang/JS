import DeepLearning_Module.util as util

def test(X,batch_size,ModelUser,show_progress=True):
    if show_progress:
        print("Test:")
    ModelUser.model_to_eval()

    attack_type_num = X.shape[0]
    attack_aim_type_num = X.shape[1]
    trajectory_start_point_num = X.shape[2]
    uav_num = X.shape[4]
    output = [[],[],[],[]]

    flatten_SW = util.flatten_SW(X)

    sample_num = flatten_SW.shape[0]
    batch_num = sample_num // batch_size
    if batch_num * batch_size < sample_num:
        batch_num += 1

    for b_i in range(batch_num):
        batch_X = util.array2tensor(
            flatten_SW[b_i*batch_size:(b_i+1)*batch_size]
        ).to(ModelUser.device)
        if show_progress:
            print("\tbatch:[{}/{}] sample_num:[{}]".format(b_i+1,batch_num,batch_X.shape[0]))

        ModelUser.forward(batch_X)
        util.out_GPU(batch_X)

        pre_label1,pre_label2,pre_label3,pre_compensation = util.predict2array(
            ModelUser.pre_label1,ModelUser.pre_label2,
            ModelUser.pre_label3,ModelUser.pre_compensation
        )
        util.collect_output(
            output,ModelUser.alpha,
            pre_label1,pre_label2,
            pre_label3,pre_compensation
        )

    util.output2array(output)
    util.reshape_output(
        output,
        attack_type_num,attack_aim_type_num,
        trajectory_start_point_num,uav_num
    )
    return output