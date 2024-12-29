import torch
import torch.nn as nn
import torch.nn.functional as F

# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from thop import profile
# torch.manual_seed(99)

class convBlock(nn.Module):
    def __init__(self, out_channel):
        super(convBlock, self).__init__()
        self.out_channel = out_channel

        self.p1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.out_channel[0][0],
                kernel_size=(5, 9),
                stride=(1, 1),
            ),
        )

        self.p2_conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=self.out_channel[1][0],
            kernel_size=(6),
            stride=(1),
        )

        self.p2_conv2d = nn.Conv2d(
            in_channels=self.out_channel[1][0],
            out_channels=self.out_channel[1][1],
            kernel_size=(5, 7),
            stride=(1, 1),
        )

        self.p3_conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.out_channel[2][0],
                kernel_size=(6),
                stride=(1),
            ),
            nn.Conv1d(
                in_channels=self.out_channel[2][0],
                out_channels=self.out_channel[2][1],
                kernel_size=(6),
                stride=(1),
            ),
        )

        self.p3_conv2d = nn.Conv2d(
            in_channels=self.out_channel[2][1],
            out_channels=self.out_channel[2][2],
            kernel_size=(5, 2),
            stride=(1, 1),
        )

    def forward(self, x):
        batch_seqlen = x.shape[0]
        uav_num = x.shape[1]
        feature_num = x.shape[2]

        p1_x = x.unsqueeze(1)
        p1 = F.relu(self.p1(p1_x))
        p1 = p1.reshape((batch_seqlen, -1))

        x_conv1d = x.reshape((batch_seqlen * uav_num, feature_num)).unsqueeze(1)

        p2_x_conv2d = F.relu(self.p2_conv1d(x_conv1d))
        p2_x_conv2d = p2_x_conv2d.reshape((batch_seqlen, uav_num, self.out_channel[1][0], -1))
        p2_x_conv2d = p2_x_conv2d.swapaxes(1, 2)
        p2 = F.relu(self.p2_conv2d(p2_x_conv2d))
        p2 = p2.reshape((batch_seqlen, -1))

        p3_x_conv1d = F.relu(self.p3_conv1d(x_conv1d))
        p3_x_conv2d = p3_x_conv1d.reshape((batch_seqlen, uav_num, self.out_channel[2][0], -1))
        p3_x_conv2d = p3_x_conv2d.swapaxes(1, 2)
        p3 = F.relu(self.p3_conv2d(p3_x_conv2d))
        p3 = p3.reshape((batch_seqlen, -1))
        return torch.concat([x.reshape((batch_seqlen, -1)), p1, p2, p3], dim=1)

# 参数：
# in_channel:int 输入数据的通道数
# cp1:list[int] 关于线路1的相关配置
# cp2:list[int] 关于线路2的相关配置
# cp3:list[int] 关于线路3的相关配置
# cp4:list[int] 关于线路4的相关配置
# 线路配置：
# 参数1:
class inception_1D(nn.Module):
    def __init__(self, input_channel, cp1, cp2, cp3, cp4):
        super(inception_1D, self).__init__()

        self.p1 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp1[0],
                kernel_size=3,
                stride=2,
            ),
        )

        self.p2 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp2[0],
                kernel_size=1,
                stride=1,
            ),
            nn.Conv1d(
                in_channels=cp2[0],
                out_channels=cp2[1],
                kernel_size=3,
                stride=2,
                dilation=2,
                padding=1,
            ),
        )

        self.p3 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp3[0],
                kernel_size=1,
                stride=1,
            ),
            nn.Conv1d(
                in_channels=cp3[0],
                out_channels=cp3[1],
                kernel_size=5,
                stride=2,
                padding=1,
            ),
        )

        self.p4 = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=2,
                stride=1,
            ),
            nn.Conv1d(
                input_channel,
                out_channels=cp4[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        # len(shape) == 3 表示此时第一维是batch
        if len(p1.shape) == 3:
            return F.relu(torch.concat([p1, p2, p3, p4], dim=1))
        # len(shape) == 2 表示第一维是channel，且不存在batch维
        elif len(p1.shape) == 2:
            return F.relu(torch.concat([p1, p2, p3, p4], dim=0))
        return None

class inception_2D(nn.Module):
    def __init__(self, input_channel, cp1, cp2, cp3, cp4):
        super(inception_2D, self).__init__()

        self.p1 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp1[0],
                kernel_size=8,
                stride=16,
            ),
        )

        self.p2 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp2[0],
                kernel_size=4,
                stride=4,
            ),
            nn.Conv1d(
                in_channels=cp2[0],
                out_channels=cp2[1],
                kernel_size=6,
                stride=4,
                dilation=1,
                padding=1,
            ),
        )

        self.p3 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp3[0],
                kernel_size=8,
                stride=4,
            ),
            nn.Conv1d(
                in_channels=cp3[0],
                out_channels=cp3[1],
                kernel_size=4,
                stride=4,
                padding=1,
            ),
        )

        self.p4 = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=3,
                stride=4,
            ),
            nn.Conv1d(
                input_channel,
                out_channels=cp4[0],
                kernel_size=10,
                stride=4,
                padding=3,
            ),
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        # len(shape) == 3 表示此时第一维是batch
        if len(p1.shape) == 3:
            return F.relu(torch.concat([p1, p2, p3, p4], dim=1))
        # len(shape) == 2 表示第一维是channel，且不存在batch维
        elif len(p1.shape) == 2:
            return F.relu(torch.concat([p1, p2, p3, p4], dim=0))
        return None

class FeatureMining(nn.Module):
    def __init__(self, hidden_size=256, layer=3, dropout=0):
        super(FeatureMining, self).__init__()

        self.cB = convBlock(
            [[2],
             [2, 2],
             [3, 3, 3], ]
        )

        # self.input_size随self.finalConv的输出维度变化
        self.input_size = 1089 # 5:109  10:354  15:599 20:844 25:1089
        self.hidden_size = hidden_size
        self.base_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=layer,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        # x:[batch,seqlen,uav_num,feature_num]
        batch = x.shape[0]
        seqlen = x.shape[1]

        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], -1))
        lstm_x = self.cB(x)
        lstm_x = lstm_x.reshape((batch, seqlen, -1))
        lstm_y = self.base_lstm(lstm_x)
        return lstm_y

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.input_size = 75
        self.output_size = 1024
        self.attention_n = 8
        self.hidden_size = 1024
        self.embedding = nn.Linear(self.input_size,self.output_size)
        # encoder参数量计算公式：
        # 4*d_model^2*nhead + 2*d_model*dim_feedforward + 2*d_model
        self.e_layer = nn.TransformerEncoderLayer(
            d_model=self.output_size,
            nhead=self.attention_n,
            dim_feedforward=self.hidden_size,
        )

    def forward(self,x):
        batch = x.shape[0]
        seqlen = x.shape[1]
        # encoder_x = x.reshape(batch,seqlen,-1)
        embedding_x = x.reshape(batch,seqlen,-1)
        embedding_y = self.embedding(embedding_x)
        encoder_x = embedding_y.reshape(batch,seqlen,-1)
        y = self.e_layer(encoder_x)
        return y

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        self.input_channel = 3
        self.conv_kernel_size = 3
        self.pool_kernel_size = 2
        self.layer_kernel_num = [64,128,256,512,512]
        self.layer_conv_num = [2,2,3,3,3]
        self.block0 = nn.Sequential(
            nn.Conv2d(self.input_channel,self.layer_kernel_num[0],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[0],self.layer_kernel_num[0],
                      kernel_size=self.conv_kernel_size,padding=1),
            # nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=1),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(self.layer_kernel_num[0],self.layer_kernel_num[1],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[1],self.layer_kernel_num[1],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.layer_kernel_num[1],self.layer_kernel_num[2],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[2],self.layer_kernel_num[2],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[2],self.layer_kernel_num[2],
                      kernel_size=1),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=1),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(self.layer_kernel_num[2],self.layer_kernel_num[3],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[3],self.layer_kernel_num[3],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[3],self.layer_kernel_num[3],
                      kernel_size=1),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=(1)),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(self.layer_kernel_num[3],self.layer_kernel_num[4],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[4],self.layer_kernel_num[4],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[4],self.layer_kernel_num[4],
                      kernel_size=1),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=(1,2)),
        )

    def forward(self,x):
        batch = x.shape[0]
        block0_y = self.block0(x)
        block1_y = self.block1(block0_y)
        block2_y = self.block2(block1_y)
        block3_y = self.block3(block2_y)
        block4_y = self.block4(block3_y)
        y = block4_y.transpose(1,3)
        y = y.reshape(batch,3,-1)
        return y

class MultiTask(nn.Module):
    def __init__(self, input_channel, uav_num, dropout=(0,0,0,0)):
        super(MultiTask, self).__init__()

        self.uav_num = uav_num

        self.block1 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                4,
                kernel_size=1,
                stride=1,
            ),
            # nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            inception_1D(4, (4,), (4, 8), (4, 6), (6,)),
        )
        # 判断是否受到攻击
        # 通过linear产生2*uav_num长度的向量
        # 后reshape到(uav_num,2)对每一行进行softmax
        self.output1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1),
            nn.Flatten(),
            # nn.Dropout(p=dropout[0]),
            nn.Linear(4032, self.uav_num), # 128:432 1024:4032
        )

        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            inception_1D(24, (24,), (36, 48), (28, 32), (32,)),
        )
        # 判断是否受到何种攻击
        # 通过linear产生2*uav_num长度的向量
        # 后reshape到(uav_num,2)对每一行进行softmax
        self.output2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1),
            nn.Flatten(),
            # nn.Dropout(p=dropout[1]),
            nn.Linear(11016, self.uav_num), # 128:816 1024:11016
        )

        self.block3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            inception_1D(24, (24,), (36, 48), (28, 32), (32,)),
        )
        # 判断是否受到何种攻击
        # 通过linear产生2*uav_num长度的向量
        # 后reshape到(uav_num,2)对每一行进行softmax
        self.output3 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1),
            nn.Flatten(),
            # nn.Dropout(p=dropout[2]),
            nn.Linear(11016, self.uav_num), # 128:816 1024:11016
        )

        self.block4 = nn.Sequential(
            inception_1D(136, (136,), (192, 256), (168, 192), (168,)),
            # nn.BatchNorm1d(752),
            nn.ReLU(),
        )
        # 回归控制信号攻击的补偿结果
        self.output4 = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(p=dropout[3]),
        )

        output4_input_size = 30832 # 128:2256 1024:30832
        command_template = "self.output4_{output4_id} = nn.Linear(output4_input_size, 2)"
        for i in range(self.uav_num):
            exec(command_template.format(output4_id=i))

    def forward(self, x):
        # 规定模型的输入必须包含batch维
        # 即，[batch, 2, lstm_hidden_size]
        x_b1 = self.block1(x)
        x1 = self.output1(x_b1)
        x1 = torch.reshape(x1, (x.shape[0], self.uav_num))
        out1 = F.sigmoid(x1)  # 是否受到攻击

        x_b2 = self.block2(x_b1)
        x2 = self.output2(x_b2)
        x2 = torch.reshape(x2, (x2.shape[0], self.uav_num))
        out2 = F.sigmoid(x2)  # 是否受到state攻击

        x_b3 = self.block3(x_b1)
        x3 = self.output3(x_b3)
        x3 = torch.reshape(x3, (x3.shape[0], self.uav_num))
        out3 = F.sigmoid(x3)  # 是否受到control攻击

        x_b4 = self.block4(x_b3)
        out4 = self.output4(x_b4)  # 补偿结果
        sub_command_template = "self.output4_{id}(out4)"
        command_tempalte = "[" + ",".join([sub_command_template.format(id=i) for i in range(self.uav_num)]) + "]"
        control = eval(command_tempalte)
        return out1, out2, out3, torch.stack(control, dim=1)