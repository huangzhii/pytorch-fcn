import os.path as osp

import fcn
import torch.nn as nn

from .fcn32s import get_upsampling_weight

def happyprint(string, obj):
    # print(string, obj)
    return

class FCN8s(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return
        # return fcn.data.cached_download(
        #     url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
        #     path=cls.pretrained_model,
        #     md5='dbd9bbb3829a3184913bccc74373afbb',
        # )

    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv3d(1, 8, 3, padding=90)
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv3d(8, 8, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv3d(8, 16, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv3d(64, 512, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout3d()

        # fc7
        self.fc7 = nn.Conv3d(512, 512, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout3d()

        self.score_fr = nn.Conv3d(512, n_class, 1)
        self.score_pool3 = nn.Conv3d(32, n_class, 1)
        self.score_pool4 = nn.Conv3d(64, n_class, 1)

        self.upscore2 = nn.ConvTranspose3d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose3d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose3d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    # def __init__(self, n_class=21):
    #     super(FCN8s, self).__init__()
    #     # conv1
    #     self.conv1_1 = nn.Conv3d(1, 64, 1, padding=100)
    #     self.relu1_1 = nn.ReLU(inplace=True)
    #     self.conv1_2 = nn.Conv3d(64, 64, 1, padding=1)
    #     self.relu1_2 = nn.ReLU(inplace=True)
    #     self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2

    #     # conv2
    #     self.conv2_1 = nn.Conv3d(64, 128, 1, padding=1)
    #     self.relu2_1 = nn.ReLU(inplace=True)
    #     self.conv2_2 = nn.Conv3d(128, 128, 1, padding=1)
    #     self.relu2_2 = nn.ReLU(inplace=True)
    #     self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4

    #     # conv3
    #     self.conv3_1 = nn.Conv3d(128, 256, 1, padding=1)
    #     self.relu3_1 = nn.ReLU(inplace=True)
    #     self.conv3_2 = nn.Conv3d(256, 256, 1, padding=1)
    #     self.relu3_2 = nn.ReLU(inplace=True)
    #     self.conv3_3 = nn.Conv3d(256, 256, 1, padding=1)
    #     self.relu3_3 = nn.ReLU(inplace=True)
    #     self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/8

    #     # conv4
    #     self.conv4_1 = nn.Conv3d(256, 512, 1, padding=1)
    #     self.relu4_1 = nn.ReLU(inplace=True)
    #     self.conv4_2 = nn.Conv3d(512, 512, 1, padding=1)
    #     self.relu4_2 = nn.ReLU(inplace=True)
    #     self.conv4_3 = nn.Conv3d(512, 512, 1, padding=1)
    #     self.relu4_3 = nn.ReLU(inplace=True)
    #     self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/16

    #     # conv5
    #     self.conv5_1 = nn.Conv3d(512, 512, 1, padding=1)
    #     self.relu5_1 = nn.ReLU(inplace=True)
    #     self.conv5_2 = nn.Conv3d(512, 512, 1, padding=1)
    #     self.relu5_2 = nn.ReLU(inplace=True)
    #     self.conv5_3 = nn.Conv3d(512, 512, 1, padding=1)
    #     self.relu5_3 = nn.ReLU(inplace=True)
    #     self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/32

    #     # fc6
    #     self.fc6 = nn.Conv3d(512, 4096, 7)
    #     self.relu6 = nn.ReLU(inplace=True)
    #     self.drop6 = nn.Dropout3d()

    #     # fc7
    #     self.fc7 = nn.Conv3d(4096, 4096, 1)
    #     self.relu7 = nn.ReLU(inplace=True)
    #     self.drop7 = nn.Dropout3d()

    #     self.score_fr = nn.Conv3d(4096, n_class, 1)
    #     self.score_pool3 = nn.Conv3d(256, n_class, 1)
    #     self.score_pool4 = nn.Conv3d(512, n_class, 1)

    #     self.upscore2 = nn.ConvTranspose3d(
    #         n_class, n_class, 4, stride=2, bias=False)
    #     self.upscore8 = nn.ConvTranspose3d(
    #         n_class, n_class, 16, stride=8, bias=False)
    #     self.upscore_pool4 = nn.ConvTranspose3d(
    #         n_class, n_class, 4, stride=2, bias=False)

    #     self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose3d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)


class FCN8sAtOnce(FCN8s):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    def forward(self, x):
        h = x
        happyprint("init: ", x.data[0].shape)

        h = self.relu1_1(self.conv1_1(h))
        happyprint("after conv1_1: ", h.data[0].shape)
        
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        happyprint("after pool1: ", h.data[0].shape)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        happyprint("after pool2: ", h.data[0].shape)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        happyprint("after pool3: ", h.data[0].shape)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        happyprint("after pool4: ", h.data[0].shape)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        happyprint("after pool5: ", h.data[0].shape)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        happyprint("after score_fr: ", h.data[0].shape)
        h = self.upscore2(h)

        happyprint("after upscore2: ", h.data[0].shape)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        happyprint("after score_pool4: ", h.data[0].shape)

        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3], 5:5 + upscore2.size()[4]]

        score_pool4c = h  # 1/16
        happyprint("after score_pool4c: ", h.data[0].shape)

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        happyprint("after upscore_pool4: ", h.data[0].shape)

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
             9:9 + upscore_pool4.size()[2],
             9:9 + upscore_pool4.size()[3],
             9:9 + upscore_pool4.size()[4]]
        score_pool3c = h  # 1/8
        happyprint("after score_pool3: ", h.data[0].shape)

        # print(upscore_pool4.data[0].shape)
        # print(score_pool3c.data[0].shape)

        # Adjusting stride in self.upscore2 and self.upscore_pool4
        # and self.conv1_1 can change the tensor shape (size).
        # I don't know why!

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h) # dim: 88^3
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3], 31:31 + x.size()[4]].contiguous()
        happyprint("after upscore8: ", h.data[0].shape)
        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv3d) and isinstance(l2, nn.Conv3d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
