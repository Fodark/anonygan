import torch
import torch.nn as nn
from einops import rearrange


class GCN(nn.Module):
    """Graph convolution unit (single layer)"""

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        h = rearrange(x, "b s n -> b n s")
        h = self.conv1(h)
        # (n, num_node, num_state) -> (n, num_state, num_node)
        h = rearrange(h, "b n s -> b s n")
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit_2D(nn.Module):
    def __init__(self, num_input, num_mid, normalize=False):
        super().__init__()

        self.normalize = normalize
        self.num_states = 2 * num_mid
        self.num_nodes = num_mid

        # reduce dim
        self.conv_state = nn.Conv2d(num_input, self.num_states, kernel_size=1)
        # projection map
        self.conv_proj = nn.Conv2d(num_input, self.num_nodes, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn1 = GCN(num_state=self.num_states, num_node=self.num_nodes)
        self.gcn2 = GCN(num_state=self.num_states, num_node=self.num_nodes)
        # ----------
        # extend dimension
        self.conv_extend = nn.Conv2d(
            self.num_states, num_input, kernel_size=1, bias=False
        )

        self.blocker = nn.BatchNorm2d(
            num_input, eps=1e-04
        )  # should be zero initialized

    def forward(self, x):
        """
        :param x: (n, c, d, h, w)
        """
        b, c, h, w = x.shape
        # print("b", b, "c", c, "h", h, "w", w)

        # n = x.size(0)
        #         c = torch.div(x.size(1),2).item()
        c_half = c // 2

        x1 = x[:, :c_half]
        x2 = x[:, c_half:]

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped1 = self.conv_state(x1)
        x_state_reshaped1 = rearrange(x_state_reshaped1, "b c h w -> b c (h w)")

        x_state_reshaped2 = self.conv_state(x2)
        x_state_reshaped2 = rearrange(x_state_reshaped2, "b c h w -> b c (h w)")

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped1 = self.conv_proj(x2)
        x_proj_reshaped1 = rearrange(x_proj_reshaped1, "b c h w -> b c (h w)")

        x_proj_reshaped2 = self.conv_proj(x1)
        x_proj_reshaped2 = rearrange(x_proj_reshaped2, "b c h w -> b c (h w)")

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped1 = x_proj_reshaped1
        x_rproj_reshaped2 = x_proj_reshaped2

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state1 = torch.einsum(
            "b s d, b n d -> b s n", x_state_reshaped1, x_proj_reshaped1
        )
        x_n_state2 = torch.einsum(
            "b s d, b n d -> b s n", x_state_reshaped2, x_proj_reshaped2
        )

        if self.normalize:
            print("using normalize")
            x_n_state1 = x_n_state1 * (1.0 / x_state_reshaped1.size(2))
            x_n_state2 = x_n_state2 * (1.0 / x_state_reshaped2.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel1 = self.gcn1(x_n_state1)
        x_n_rel2 = self.gcn2(x_n_state2)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped1 = torch.einsum(
            "b s n, b n d -> b s d", x_n_rel1, x_rproj_reshaped1
        )
        x_state_reshaped2 = torch.einsum(
            "b s n, b n d -> b s d", x_n_rel2, x_rproj_reshaped2
        )

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state1 = rearrange(x_state_reshaped1, "b s (h w) -> b s h w", h=h, w=w)
        x_state2 = rearrange(x_state_reshaped2, "b s (h w) -> b s h w", h=h, w=w)

        # (n, num_state, h, w) -> (n, num_in, h, w)
        out1 = x1 + self.blocker(self.conv_extend(x_state1))
        out2 = x2 + self.blocker(self.conv_extend(x_state2))

        return torch.cat((out1, out2), 1)


class GraphBlock(nn.Module):
    def __init__(self, dim, use_bias, cated_stream2=False):
        super(GraphBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, use_bias, cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(
            dim, use_bias, cal_att=True, cated_stream2=cated_stream2
        )
        self.gcn = GloRe_Unit_2D(128, 64)

    def build_conv_block(self, dim, use_bias, cated_stream2=False, cal_att=False):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]

        if cated_stream2:
            conv_block += [
                nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=0, bias=use_bias),
                nn.BatchNorm2d(dim * 2),
                nn.ReLU(True),
            ]
        else:
            conv_block += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
            ]
        conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(1)]

        if cal_att:
            if cated_stream2:
                conv_block += [
                    nn.Conv2d(dim * 2, dim, kernel_size=3, padding=0, bias=use_bias)
                ]
            else:
                conv_block += [
                    nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
                ]
        else:
            conv_block += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                nn.BatchNorm2d(dim),
            ]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        # print("x1", x1.shape)
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        x2_out = self.gcn(x2_out)

        att = torch.sigmoid(x2_out)
        # print("att", att.shape, "x1", x1_out.shape)
        x1_out = x1_out * att
        out = x1 + x1_out  # residual connection
        # stream2 receive feedback from stream1
        x2_out = torch.cat((x2_out, out), 1)
        return out, x2_out
