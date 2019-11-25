import torch
import torch.nn as nn
import imp
import torchvision
from torchvision.models import vgg19
from network.model import Cropped_VGG19
from network.vgg import *
import torch.nn.functional as F


class LossCnt(nn.Module):
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device):
        super(LossCnt, self).__init__()

        self.vgg19_layers = [1, 6, 11, 18, 25]
        self.vggface_layers = [1, 6, 11, 20, 29]

        self.VGG19 = vgg19(pretrained=True)
        self.VGG19.eval()
        self.VGG19.to(device)

        self.VGGFace = vgg_face(pretrained=True)
        self.VGGFace.eval()
        self.VGGFace.to(device)


        self.VGG19_Activations = VGG_Activations(self.VGG19, self.vgg19_layers)
        self.VGGface_Activations = VGG_Activations(self.VGGFace, self.vggface_layers)


    def forward(self, x, x_hat, vgg19_weight=1e-2, vggface_weight=2e-3):

        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(x.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(x.device)

        x = (x - IMG_NET_MEAN) / IMG_NET_STD
        x_hat = (x_hat - IMG_NET_MEAN) / IMG_NET_STD

        vgg19_x_activations = self.VGG19_Activations(x)
        vgg19_x_hat_activations = self.VGG19_Activations(x_hat)

        vgg19_loss = 0
        for i in range(len(self.vgg19_layers)):
            vgg19_loss += F.l1_loss(vgg19_x_activations[i], vgg19_x_hat_activations[i])

        vggface_x_activations = self.VGGface_Activations(x)
        vggface_x_hat_activations = self.VGGface_Activations(x_hat)

        vggface_loss = 0
        for i in range(len(self.vggface_layers)):
            vggface_loss += F.l1_loss(vggface_x_activations[i], vggface_x_hat_activations[i])

        loss = vgg19_loss*vgg19_weight + vggface_loss*vggface_weight
        return loss


class LossAdv(nn.Module):
    def __init__(self, FM_weight=1e1):
        super(LossAdv, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.FM_weight = FM_weight
        
    def forward(self, r_hat, D_res_list, D_hat_res_list):
        lossFM = 0
        for res, res_hat in zip(D_res_list, D_hat_res_list):
            lossFM += self.l1_loss(res, res_hat)
            
        return -r_hat.squeeze().mean() + lossFM * self.FM_weight


class LossMatch(nn.Module):
    def __init__(self, device, match_weight=8e1):
        super(LossMatch, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.match_weight = match_weight
        self.device = device
        
    def forward(self, e_vectors, W, i):
        loss = torch.zeros(e_vectors.shape[0],1).to(self.device)
        for b in range(e_vectors.shape[0]):
            for k in range(e_vectors.shape[1]):
                loss[b] += torch.abs(e_vectors[b,k].squeeze() - W[:,i[b]]).mean()
            loss[b] = loss[b]/e_vectors.shape[1]
        loss = loss.mean()
        return loss * self.match_weight
    
class LossG(nn.Module):
    """
    Loss for generator meta training
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device, vgg19_weight=1e-2, vggface_weight=2e-3):
        super(LossG, self).__init__()
        
        self.LossCnt = LossCnt(VGGFace_body_path, VGGFace_weight_path, device)
        self.lossAdv = LossAdv()
        self.lossMatch = LossMatch(device=device)
        
    def forward(self, x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, W, i):
        loss_cnt = self.LossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        loss_match = self.lossMatch(e_vectors, W, i)
        return loss_cnt + loss_adv + loss_match

class LossGF(nn.Module):
    """
    Loss for generator finetuning
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device, vgg19_weight=1e-2, vggface_weight=2e-3):
        super(LossGF, self).__init__()
        
        self.LossCnt = LossCnt(VGGFace_body_path, VGGFace_weight_path, device)
        self.lossAdv = LossAdv()
        
    def forward(self, x, x_hat, r_hat, D_res_list, D_hat_res_list):
        loss_cnt = self.LossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        return loss_cnt + loss_adv