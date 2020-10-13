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
import robust_test as robust_test
from utils import *

#   default settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Auxiliary Learning')
parser.add_argument('--epoch', default=210, type=int)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--beta', default=0.05, type=float)
parser.add_argument('--gama', default=5.0, type=float)
parser.add_argument('--rotate', default=20, type=float)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--noise', default=10, type=float)
parser.add_argument('--blur', default=0.5, type=float)
parser.add_argument('--width', default=64, type=int)
args = parser.parse_args()
BATCH_SIZE = 128
LR = 0.1

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

trans_contrast = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_noise = transforms.Compose([
    transforms.ToPILImage(),
    GaussianNoise(20),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_blur = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda x: x.filter(PIL.ImageFilter.GaussianBlur(1.0))),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_clean = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trans_rotate = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(
    root='/home2/lthpc/data',
    train=True,
    download=False,
    transform=transform_train
)
testset = torchvision.datasets.CIFAR100(
    root='/home2/lthpc/data',
    train=False,
    download=False,
    transform=transform_test
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

net = None
if args.depth == 18:
    net = models.resnet18()
if args.depth == 50:
    net = models.resnet50()
if args.depth == 101:
    net = models.resnet101()
if args.depth == 152:
    net = models.resnet152()

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    for epoch in range(args.epoch):
        if epoch in [60, 140, 180]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        weight_loss_all = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs, labels.cuda()
            loaders = [trans_clean, trans_blur, trans_noise, trans_contrast, trans_rotate]
            auxiliary_inputs = get_auxiliary_data(inputs, loaders)
            outputs = net(auxiliary_inputs)
            batchsize = labels.size(0)
            output_clean, outputs_blur, outputs_noise, outputs_contrast, outputs_rotate = \
                outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

            loss = criterion(output_clean, labels)

            loss += criterion(outputs_noise, labels) * args.alpha
            loss += criterion(outputs_blur, labels) * args.alpha
            loss += criterion(outputs_contrast, labels) * args.alpha
            loss += criterion(outputs_rotate, labels) * args.alpha

            output_clean_detach = output_clean.detach()
            loss += CrossEntropy(outputs_noise, output_clean_detach) * args.beta
            loss += CrossEntropy(outputs_blur, output_clean_detach) * args.beta
            loss += CrossEntropy(outputs_contrast, output_clean_detach) * args.beta
            loss += CrossEntropy(outputs_rotate, output_clean_detach) * args.beta

            fc_para = list(net.auxiliary_classifiers_list[0].fc.parameters())
            fc1_w, fc1_b = fc_para[0], fc_para[1]

            fc_para = list(net.auxiliary_classifiers_list[1].fc.parameters())
            fc2_w, fc2_b = fc_para[0], fc_para[1]

            fc_para = list(net.auxiliary_classifiers_list[2].fc.parameters())
            fc3_w, fc3_b = fc_para[0], fc_para[1]

            fc_para = list(net.auxiliary_classifiers_list[3].fc.parameters())
            fc4_w, fc4_b = fc_para[0], fc_para[1]

            fc_para = list(net.auxiliary_classifiers_list[4].fc.parameters())
            fc5_w, fc5_b = fc_para[0], fc_para[1]

            weight_loss = ((fc1_w - fc2_w) ** 2).sum() + ((fc1_b - fc2_b) ** 2).sum()
            weight_loss += ((fc1_w - fc3_w) ** 2).sum() + ((fc1_b - fc3_b) ** 2).sum()
            weight_loss += ((fc1_w - fc4_w) ** 2).sum() + ((fc1_b - fc4_b) ** 2).sum()
            weight_loss += ((fc1_w - fc5_w) ** 2).sum() + ((fc1_b - fc5_b) ** 2).sum()
            if epoch >= 180:
                loss += weight_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            weight_loss_all += weight_loss.item()
            _, predicted = torch.max(output_clean.data, 1)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels.data).cpu().sum())
            print('[epoch:%d, iter:%d] Loss: %.03f Weight Loss: %0.3f | Acc: %.4f%%'
                % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), weight_loss_all / (i+1), 100 * correct / total))
        if epoch >= 190 or epoch % 10 == 0:
            robust_test.robust_test(net)

        print("Waiting Test!")
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                images4 = [images]
                outputs = net(images4)
                outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                total += float(labels.size(0))
                correct += float((predicted == labels).sum())
            acc1 = (100 * correct/total)
        print('Test Set Accuracy: %.4f%%' % (acc1))
    print("Training Finished, TotalEPOCH=%d" % args.epoch)


