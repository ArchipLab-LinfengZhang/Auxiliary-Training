import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
import models
import numpy as np
from PIL import Image
import PIL


class GaussianNoise(object):
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, pic):
        arra = np.array(pic)
        noises = np.random.normal(0, self.strength, arra.shape)
        noises = np.uint8(noises)
        arra += noises
        pic = Image.fromarray(arra)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def get_auxiliary_data(data, loaders):
    data_list = [[] for _ in loaders]
    for index_sample in range(data.size(0)):
        sample = data[index_sample]
        for index_loader in range(len(loaders)):
            trans_sample = loaders[index_loader](sample)
            data_list[index_loader].append(torch.unsqueeze(trans_sample, dim=0))

    for index_loader in range(len(loaders)):
        data_list[index_loader] = torch.cat(data_list[index_loader], dim=0)
        data_list[index_loader] = data_list[index_loader].cuda()
    return data_list
