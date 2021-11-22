import torch
import torch.nn as nn

from .eca_layer import eca_layer
from .utils_modules_new import *


class Generator(nn.Module):
    def __init__(self, ngf=64, n_blocks=6, n_downsampling=2, ch_input=6, use_ch_att=True, reduced_landmarks=False):
        super(Generator, self).__init__()
        
        self.input_nc_s1 = ch_input
        if reduced_landmarks:
            self.input_nc_s2 = 2 * 29
        else:
            self.input_nc_s2 = 2 * 68
        self.output_nc = 3
        self.ngf = ngf
        self.use_ch_att = use_ch_att
        
        if use_ch_att:
            self.ch_att = eca_layer(2 * 68)

        # down_sample
        model_stream1_down = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        ]

        model_stream2_down = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        ]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_stream1_down += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
            model_stream2_down += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # att_block in place of res_block
        mult = 2 ** n_downsampling
        cated_stream2 = [True for i in range(n_blocks)]
        cated_stream2[0] = False
        attBlock = nn.ModuleList()
        for i in range(n_blocks):
            attBlock.append(
                GraphBlock(ngf * mult, use_bias=False, cated_stream2=cated_stream2[i])
                # GraphBlock(
                #     ngf * mult,
                #     use_bias=False,
                #     cated_stream2=cated_stream2[i],
                #     padding_type="reflect",
                #     norm_layer=nn.BatchNorm2d,
                #     use_dropout=True,
                # )
            )

        # up_sample
        model_stream1_up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_stream1_up += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model_stream1_up += [nn.ReflectionPad2d(3)]
        model_stream1_up += [nn.Conv2d(ngf, self.output_nc, kernel_size=7, padding=0)]
        model_stream1_up += [nn.Tanh()]

        # self.model = nn.Sequential(*model)
        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        # self.att = nn.Sequential(*attBlock)
        self.att = attBlock
        self.stream1_up = nn.Sequential(*model_stream1_up)

        self.conv_att1 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.conv_att2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        # self.conv_att3 = nn.ConvTranspose2d(
        #     64, 2, kernel_size=1, stride=1, padding=0, bias=False
        # )
        self.conv_att3 = nn.ConvTranspose2d(
            64, 1, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, input):  # x from stream 1 and stream 2
        # here x should be a tuple
        input_image, x2 = input

        if self.use_ch_att:
            x2 = self.ch_att(x2)
        
        # print('I', input_image.shape, 'P', x2.shape)

        x1 = self.stream1_down(input_image)
        x2 = self.stream2_down(x2)

        for model in self.att:
            x1, x2 = model(x1, x2)

        out = self.stream1_up(x1)

        att = self.conv_att1(x1)
        att = self.conv_att2(att)
        att = torch.sigmoid(self.conv_att3(att))

        # att1 = att[:, 0:1, :, :]
        # att2 = att[:, 1:2, :, :]
        att = att.repeat(1, 3, 1, 1)
        # att2 = att2.repeat(1, 3, 1, 1)

        if self.input_nc_s1 == 6:
            out = out * att + input_image[:, 3:] * (1 - att)
        else:
            out = out * att + input_image * (1 - att)
        return torch.clamp(out, -1, 1)  # torch.tanh(out)
