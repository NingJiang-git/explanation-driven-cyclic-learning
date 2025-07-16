import torch
from PIL import Image
import sys
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F
from scipy.io import savemat

def grad_rollout(attentions, gradients, discard_ratio):
    gradients.reverse()
    attentions = attentions[0:22]
    result = torch.zeros(224,176)

    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0 

            b,w,h = attention_heads_fused.size(0), attention_heads_fused.size(1), attention_heads_fused.size(2)
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            flat = flat.float()
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            flat[0, indices] = 0

            attention_heads_fused = flat.reshape(b,w,h)
            attention_heads_fused = F.interpolate(attention_heads_fused.unsqueeze(dim=1).float(), size=[224, 176], mode='nearest')
            result = attention_heads_fused + result
            result = normalize(result)

    mask = result
    return mask    

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())



    def __call__(self, input_tensor, target_label):
        self.model.zero_grad()
        output = self.model(input_tensor, input_tensor)
        loss_function = torch.nn.MultiLabelSoftMarginLoss()
        loss = loss_function(output, target_label)
        loss.backward()
        mask = grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)
        self.attentions = []
        self.attention_gradients = []
        torch.cuda.empty_cache()
        return mask
    


def normalize(X):    
    X_min, index = torch.min(X, dim=0, keepdim=True)   
    X_max, index = torch.max(X, dim=0, keepdim=True)    
    X_norm = (X - X_min) / (X_max - X_min + 1e-15) 
    return X_norm
