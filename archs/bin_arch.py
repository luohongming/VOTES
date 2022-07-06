import torch
import torch.nn as nn

from numpy.random import normal
from numpy.linalg import svd
import torch.nn.init as weight_init
import torch.nn.functional as F
from utils.registry import ARCH_REGISTRY



class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, forget_bias=1.0, kernel_size=3, padding=3//2):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size * hidden_size, 4 * hidden_size, kernel_size, padding=padding, bias=True)
        self._forget_bias = forget_bias
        self._initialize_weights()

    def _initialize_weights(self):  # The Xavier method of initialize
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                count += 1
                weight_init.xavier_uniform_(m.weight.data)
                # weight_init.kaiming_uniform(m.weight.data, a = 0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input_, prev_state):

        # get batch and spatial sizes.  input_size: B, C, H, W
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if input_.is_cuda:
                prev_state = [
                    torch.cuda.FloatTensor().resize_(state_size).zero_(),
                    torch.cuda.FloatTensor().resize_(state_size).zero_(),
                ]
            else:
                prev_state = [
                    torch.zeros(state_size),
                    torch.zeros(state_size)
                ]

        c, h = prev_state   #  c: long-term memory, h: short-term memory

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, h), 1)
        concat = self.Gates(stacked_inputs)
        i, j, f, o = concat.chunk(4, 1)

        new_c = (c * F.sigmoid(f + self._forget_bias) + F.sigmoid(i).mul(F.tanh(j)))
        new_h = F.tanh(new_c).mul(F.sigmoid(o))

        return new_h, [new_c, new_h]

def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = torch.Tensor(1, 3, 12, 12)
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=kSize//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growGate0, growGate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growGate0
        G = growGate
        C = nConvLayers
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))

        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x    # input G0, output G0.


class RDN_residual_interp_x_input(nn.Module):
    def __init__(self, n_inputs, G0=64, D=6, C=4, G=32):
        super(RDN_residual_interp_x_input, self).__init__()
        kSize = 3
        inChan = 12 * n_inputs
        self.n_inputs = n_inputs
        self.D = D

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(inChan, G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growGate0=G0, growGate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, 256, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, B):
        assert len(B) == self.n_inputs
        B_shuffle = pixel_reshuffle(torch.cat(B, 1), 2)
        f_1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f_1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f_1
        B_mean = 0
        for B_i in B:
            B_mean += B_i

        F = self.UPNet(x) + B_mean / len(B)
        return F

class RDN_residual_scale2(nn.Module):
    def __init__(self, G0=64, D=6):
        super(RDN_residual_scale2, self).__init__()
        # stage 1
        self.model1_1 = RDN_residual_interp_x_input(2, G0=G0, D=D)

        # stage 2
        self.model2_1 = RDN_residual_interp_x_input(2, G0=G0, D=D)

    def forward(self, B1, B3, B5):

        self.I2_prime = self.model1_1([B1, B3])
        self.I4_prime = self.model1_1([B3, B5])

        self.I3_prime = self.model2_1([self.I2_prime, self.I4_prime])

        return self.I2_prime, self.I4_prime, self.I3_prime


class RDN_residual_scale3(nn.Module):
    def __init__(self, G0=64, D=6):
        super(RDN_residual_scale3, self).__init__()
        # stage 1
        self.model1_1 = RDN_residual_interp_x_input(2, G0=G0, D=D)

        # stage 2
        self.model2_1 = RDN_residual_interp_x_input(2, G0=G0, D=D)

        # stage 3
        self.model3_1 = RDN_residual_interp_x_input(4, G0=G0, D=D)


    def forward(self, B1, B3, B5, B7):
        self.I2_prime = self.model1_1([B1, B3])
        self.I4_prime = self.model1_1([B3, B5])
        self.I6_prime = self.model1_1([B5, B7 ])

        self.I3_prime = self.model2_1([self.I2_prime, self.I4_prime])
        self.I5_prime = self.model2_1([self.I4_prime, self.I6_prime])

        self.I4_prime_prime = self.model3_1([B3, self.I3_prime, self.I5_prime, B5])

        return self.I2_prime, self.I4_prime, self.I6_prime, self.I3_prime, self.I5_prime, self.I4_prime_prime


class RDN_residual_scale4(nn.Module):
    def __init__(self, lstm=False, G0=64, D=6):
        super(RDN_residual_scale4, self).__init__()
        # stage 1
        self.lstm = lstm
        self.model1_1 = RDN_residual_interp_x_input(2, G0=G0, D=D)
        # self.model1_2 = self.model1_1
        # self.model1_3 = self.model1_1
        # self.model1_4 = self.model1_1

        # stage 2
        if lstm == True:
            self.model2_1 = RDN_residual_interp_x_input(3, G0=G0, D=D)
        else:
            self.model2_1 = RDN_residual_interp_x_input(2, G0=G0, D=D)

        # stage 3
        if lstm == True:
            self.model3_1 = RDN_residual_interp_x_input(5, G0=G0, D=D)
        else:
            self.model3_1 = RDN_residual_interp_x_input(4, G0=G0, D=D)

        # stage 4
        if lstm == True:
            self.model4_1 = RDN_residual_interp_x_input(5, G0=G0, D=D)
        else:
            self.model4_1 = RDN_residual_interp_x_input(4, G0=G0, D=D)



    def forward(self, B1, B3, B5, B7, B9, previous_input=None):
        if self.lstm == True:
            self.I2_prime = self.model1_1([B1, B3])
            self.I4_prime = self.model1_1([B3, B5])
            self.I6_prime = self.model1_1([B5, B7])
            self.I8_prime = self.model1_1([B7, B9])
            if previous_input[0] is not None:
                p4_prime, p6_prime, p8_prime, p5_prime_prime, p7_prime_prime, p6_prime_prime_prime = previous_input
                self.I3_prime = self.model2_1([p4_prime, self.I2_prime, self.I4_prime])
                self.I5_prime = self.model2_1([p6_prime, self.I4_prime, self.I6_prime])
                self.I7_prime = self.model2_1([p8_prime, self.I6_prime, self.I8_prime])
                self.I4_prime_prime = self.model3_1([p5_prime_prime, B3, self.I3_prime, self.I5_prime, B5])
                self.I6_prime_prime = self.model3_1([p7_prime_prime, B5, self.I5_prime, self.I7_prime, B7])
                self.I5_prime_prime_prime = self.model4_1([p6_prime_prime_prime, self.I4_prime, self.I4_prime_prime, self.I6_prime_prime, self.I6_prime])

            else:
                self.I3_prime = self.model2_1([self.I2_prime, self.I2_prime, self.I4_prime])
                self.I5_prime = self.model2_1([self.I4_prime, self.I4_prime, self.I6_prime])
                self.I7_prime = self.model2_1([self.I6_prime, self.I6_prime, self.I8_prime])
                self.I4_prime_prime = self.model3_1([self.I3_prime, B3, self.I3_prime, self.I5_prime, B5])
                self.I6_prime_prime = self.model3_1([self.I5_prime, B5, self.I5_prime, self.I7_prime, B7])
                self.I5_prime_prime_prime = self.model4_1(
                    [self.I4_prime, self.I4_prime, self.I4_prime_prime, self.I6_prime_prime, self.I6_prime])

        else:
            self.I2_prime = self.model1_1([B1, B3])
            self.I4_prime = self.model1_1([B3, B5])
            self.I6_prime = self.model1_1([B5, B7])
            self.I8_prime = self.model1_1([B7, B9])
            self.I3_prime = self.model2_1([self.I2_prime, self.I4_prime])
            self.I5_prime = self.model2_1([self.I4_prime, self.I6_prime])
            self.I7_prime = self.model2_1([self.I6_prime, self.I8_prime])
            self.I4_prime_prime = self.model3_1([B3, self.I3_prime, self.I5_prime, B5])
            self.I6_prime_prime = self.model3_1([B5, self.I5_prime, self.I7_prime, B7])
            self.I5_prime_prime_prime = self.model4_1([self.I4_prime, self.I4_prime_prime, self.I6_prime_prime, self.I6_prime])


        return self.I2_prime, self.I4_prime, self.I6_prime, self.I8_prime, self.I3_prime, self.I5_prime, self.I7_prime, \
            self.I4_prime_prime, self.I6_prime_prime, self.I5_prime_prime_prime

class RDN_residual_interp_5_input_ConvLSTM_L(nn.Module):
    def __init__(self, modelType='lstm'):
        super(RDN_residual_interp_5_input_ConvLSTM_L, self).__init__()
        self.modelType = modelType
        self.clstm_4_prime = ConvLSTMCell(3, 3)
        self.clstm_6_prime = ConvLSTMCell(3, 3)
        self.clstm_8_prime = ConvLSTMCell(3, 3)
        self.clstm_5_prime_prime = ConvLSTMCell(3, 3)
        self.clstm_7_prime_prime = ConvLSTMCell(3, 3)
        self.clstm_6_prime_prime_prime = ConvLSTMCell(3, 3)
        self.model = RDN_residual_scale4(lstm=True, G0=96, D=12)
        self.prev_state = None
        self.hidden_state = None

    def forward(self, B1, B3, B5, B7, B9, B11):
        pre_state_4_prime = None
        pre_state_6_prime = None
        pre_state_8_prime = None
        pre_state_5_prime_prime = None
        pre_state_7_prime_prime = None
        pre_state_6_prime_prime_prime = None
        hidden_state_4_prime = None
        hidden_state_6_prime = None
        hidden_state_8_prime = None
        hidden_state_5_prime_prime = None
        hidden_state_7_prime_prime = None
        hidden_state_6_prime_prime_prime = None
        input1 = [B1, B3, B5, B7, B9]
        input2 = [B3, B5, B7, B9, B11]
        res = []
        for i in [input1, input2]:
            B1, B3, B5, B7, B9 = i
            previous_inputs = [hidden_state_4_prime, hidden_state_6_prime, hidden_state_8_prime,
                               hidden_state_5_prime_prime, hidden_state_7_prime_prime, hidden_state_6_prime_prime_prime]
            self.Ft_p_1 = self.model(B1, B3, B5, B7, B9, previous_inputs)  # output: I2_prime, I4_prime, I6_prime, I8_prime,
            # I3_prime, I5_prime, I7_prime, I4_prime_prime, I6_prime_prime, I5_prime_prime_prime
            hidden_state_4_prime = self.Ft_p_1[1]
            hidden_state_6_prime = self.Ft_p_1[2]
            hidden_state_8_prime = self.Ft_p_1[3]
            hidden_state_5_prime_prime = self.Ft_p_1[5]
            hidden_state_7_prime_prime = self.Ft_p_1[6]
            hidden_state_6_prime_prime_prime = self.Ft_p_1[8]
            if self.modelType == 'lstm':
                hidden_state_4_prime, pre_state_4_prime = self.clstm_4_prime(hidden_state_4_prime, pre_state_4_prime)
                hidden_state_6_prime, pre_state_6_prime = self.clstm_6_prime(hidden_state_6_prime, pre_state_6_prime)
                hidden_state_8_prime, pre_state_8_prime = self.clstm_8_prime(hidden_state_8_prime, pre_state_8_prime)
                hidden_state_5_prime_prime, pre_state_5_prime_prime = self.clstm_5_prime_prime(
                    hidden_state_5_prime_prime, pre_state_5_prime_prime)
                hidden_state_7_prime_prime, pre_state_7_prime_prime = self.clstm_7_prime_prime(
                    hidden_state_7_prime_prime, pre_state_7_prime_prime)
                hidden_state_6_prime_prime_prime, pre_state_6_prime_prime_prime = self.clstm_6_prime_prime_prime(
                    hidden_state_6_prime_prime_prime, pre_state_6_prime_prime_prime)
            else:
                pass
            res.append(self.Ft_p_1)

        return res[0][0], res[0][1], res[0][2], res[0][3], \
               res[0][4], res[0][5], res[0][6], \
               res[0][7], res[0][8], \
               res[0][9], \
               res[1][3], res[1][6], res[1][8], res[1][9]

@ARCH_REGISTRY.register()
class BIN(nn.Module):
    def __init__(self, arch_scale=2, G0=64, D=6):
        super(BIN, self).__init__()
        self.arch_scale = arch_scale
        if arch_scale == 2:
            self.model = RDN_residual_scale2(G0=G0, D=D)
        elif arch_scale == 3:
            self.model = RDN_residual_scale3()
        else:
            raise ValueError('Only scale 2 and 3 are available!')

        # self.model = RDN_residual_interp_5_input_ConvLSTM_L()

    def forward(self, x):
        # b, t, c, h, w = x.size()

        if self.arch_scale == 2:
            assert x.size(1) == 5, ("Input frames number must be 5 for scale 2!")
            out_1, out_3, out_2_prime = self.model(x[:, 0, ...], x[:, 2, ...], x[:, 4, ...])
            # out_1, out_3, out_2_prime = self.model(x[:, 1, ...], x[:, 2, ...], x[:, 3, ...])
            out1 = torch.stack([out_1, out_3], 1)
            out1_prime = out_2_prime
        elif self.arch_scale == 3:
            assert x.size(1) == 7, ("Input frames number must be 7 for scale 3!")

            out_1, out_3, out_5, out_2_prime, out_4_prime, out_5_prime = self.model(x[:, 0, ...], x[:, 2, ...], x[:, 4, ...], x[:, 6, ...])
            out1 = torch.stack([out_1, out_2_prime, out_3, out_4_prime, out_5], 1)
            out1_prime = out_5_prime


        return out1, out1_prime
