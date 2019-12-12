import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import coco,voc
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
from models.mobile_net_v3 import mobilenetv3


class MobileNetV3(nn.Module):

    def __init__(self, phase, size, head, num_classes):
        super(MobileNetV3, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        self.base = mobilenetv3()

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        loc = list()
        conf = list()

        f1, f2, f3, f4, f5, f6 = self.base(x)

        sources = [f1, f2, f3, f4, f5, f6]

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def multibox(num_classes=21):
    anchor_num = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
    loc_layers = []
    conf_layers = []

    # ===================================================================================#
    loc_layers += [nn.Conv2d(40, anchor_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(40, anchor_num[0] * num_classes, kernel_size=3, padding=1)]
    # ===================================================================================#
    loc_layers += [nn.Conv2d(112, anchor_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(112, anchor_num[1] * num_classes, kernel_size=3, padding=1)]
    # ===================================================================================#
    loc_layers += [nn.Conv2d(160, anchor_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(160, anchor_num[2] * num_classes, kernel_size=3, padding=1)]
    # ===================================================================================#
    loc_layers += [nn.Conv2d(160, anchor_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(160, anchor_num[3] * num_classes, kernel_size=3, padding=1)]
    # ===================================================================================#
    loc_layers += [nn.Conv2d(160, anchor_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(160, anchor_num[4] * num_classes, kernel_size=3, padding=1)]
    # ===================================================================================#
    loc_layers += [nn.Conv2d(160, anchor_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(160, anchor_num[5] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return

    return MobileNetV3(phase, size, multibox(num_classes), num_classes)


if __name__ == '__main__':
    x = torch.randn(1, 3, 300, 300)
    net = build_net('test')
    net.eval()
    from utils.timer import Timer

    _t = {'im_detect': Timer()}
    for i in range(300):
        _t['im_detect'].tic()
        net.forward(x)
        detect_time = _t['im_detect'].toc()
        print(detect_time)
