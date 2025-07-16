import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from generator import Restormer
from vit_grad_rollout import VITAttentionGradRollout


class Cyclic_Classfier(nn.Module):

    def __init__(self, in_chn=3, wf=16, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(Cyclic_Classfier, self).__init__()
        self.depth = depth
        self.generator = Restormer(decoder=False)

    def forward(self, x):
        out = self.generator(x, x)
        return out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class Cyclic(nn.Module):

    def __init__(self, in_chn=3, wf=16, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(Cyclic, self).__init__()
        self.depth = depth
        self.classifier = Restormer(decoder=False)
        self.classifier.load_state_dict(torch.load("/data/experiments/cyclic_CAMCAN_Class_multi_label/models/net_g_100.pth"), strict=False)
        self.generator = Restormer(decoder=True)
        self.grad_rollout = VITAttentionGradRollout(self.classifier, discard_ratio = 0.)
        

    def forward(self, x, target_label):
        self._momentum_update_classifier()
        pred_label = self.classifier(x[:,1,:,:].unsqueeze(dim=1).repeat(1,3,1,1),x)
        with torch.enable_grad():
            mask = self.grad_rollout(x[:,1,:,:].unsqueeze(dim=1).repeat(1,3,1,1), target_label)
        out = self.generator(x,mask)
        del mask
        return pred_label, out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _momentum_update_classifier(self):
        """
        Momentum update of the key encoder
        """
        for param_g, param_c in zip(
            self.generator.parameters(), self.classifier.parameters()
        ):
            param_c.data = param_c.data * 0.99 + param_g.data * 0.01

