import torch
import torch.nn as nn


class GCN(nn.Module):
    """Graph convolution unit (single layer)"""

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit(nn.Module):
    """
    Based on Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """

    def __init__(
        self,
        num_in,
        num_mid,
        ConvNd=nn.Conv3d,
        BatchNormNd=nn.BatchNorm3d,
        normalize=False,
    ):
        super(GloRe_Unit, self).__init__()

        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04)  # should be zero initialized

    def forward(self, x):
        """
        :param x: (n, c, d, h, w)
        """
        n = x.size(0)
        #         c = torch.div(x.size(1),2).item()
        c = x.size(1) // 2

        x1 = x[:, :c]
        x2 = x[:, c:]

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped1 = self.conv_state(x1).view(n, self.num_s, -1)
        x_state_reshaped2 = self.conv_state(x2).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped1 = self.conv_proj(x2).view(n, self.num_n, -1)
        x_proj_reshaped2 = self.conv_proj(x1).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped1 = x_proj_reshaped1
        x_rproj_reshaped2 = x_proj_reshaped2

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state1 = torch.matmul(x_state_reshaped1, x_proj_reshaped1.permute(0, 2, 1))
        x_n_state2 = torch.matmul(x_state_reshaped2, x_proj_reshaped2.permute(0, 2, 1))

        if self.normalize:
            print("using normalize")
            x_n_state1 = x_n_state1 * (1.0 / x_state_reshaped1.size(2))
            x_n_state2 = x_n_state2 * (1.0 / x_state_reshaped2.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel1 = self.gcn1(x_n_state1)
        x_n_rel2 = self.gcn2(x_n_state2)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped1 = torch.matmul(x_n_rel1, x_rproj_reshaped1)
        x_state_reshaped2 = torch.matmul(x_n_rel2, x_rproj_reshaped2)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state1 = x_state_reshaped1.view(n, self.num_s, *x.size()[2:])
        x_state2 = x_state_reshaped2.view(n, self.num_s, *x.size()[2:])

        # (n, num_state, h, w) -> (n, num_in, h, w)
        # out = x + self.blocker(self.conv_extend(x_state))
        out1 = x1 + self.blocker(self.conv_extend(x_state1))
        out2 = x2 + self.blocker(self.conv_extend(x_state2))

        return torch.cat((out1, out2), 1)


class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_2D, self).__init__(
            num_in,
            num_mid,
            ConvNd=nn.Conv2d,
            BatchNormNd=nn.BatchNorm2d,
            normalize=normalize,
        )


class GraphBlock(nn.Module):
    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        use_dropout,
        use_bias,
        cated_stream2=False,
        use_fuser=False,
    ):
        super(GraphBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False
        )
        self.conv_block_stream2 = self.build_conv_block(
            dim,
            padding_type,
            norm_layer,
            use_dropout,
            use_bias,
            cal_att=True,
            cated_stream2=cated_stream2,
        )

        self.gcn = GloRe_Unit_2D(128, 64)
        self.use_fuser = use_fuser

    def build_conv_block(
        self,
        dim,
        padding_type,
        norm_layer,
        use_dropout,
        use_bias,
        cated_stream2=False,
        cal_att=False,
    ):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        if cated_stream2:
            conv_block += [
                nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=p, bias=use_bias),
                norm_layer(dim * 2),
                nn.ReLU(True),
            ]
        else:
            conv_block += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                norm_layer(dim),
                nn.ReLU(True),
            ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        if cal_att:
            if cated_stream2:
                conv_block += [
                    nn.Conv2d(dim * 2, dim, kernel_size=3, padding=p, bias=use_bias)
                ]
            else:
                conv_block += [
                    nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)
                ]
        else:
            conv_block += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                norm_layer(dim),
            ]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        x2_out = self.gcn(x2_out)

        if self.use_fuser:
            mixed = torch.cat([x1_out, x2_out], dim=1)
            # x1_out, x2_out = self.transfuser(x1_out, x2_out)
            out = self.transfuser(mixed)
            x1_out, x2_out = out[:, :256, :, :], out[:, 256:, :, :]
            x1_out = x1_out + x1  # skip connection
            return x1_out, torch.cat([x2_out, x1_out], dim=1)  # feedback from x1
        else:
            att = torch.sigmoid(x2_out)
            x1_out = x1_out * att
            out = x1 + x1_out  # residual connection
            # stream2 receive feedback from stream1
            x2_out = torch.cat((x2_out, out), 1)
            return out, x2_out
