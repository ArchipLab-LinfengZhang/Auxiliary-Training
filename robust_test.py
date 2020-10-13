import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import torchvision.transforms as transforms
datas = []
for i, j, k in os.walk("./CIFARC/"):
    for item in k:
        if item[-3:] == "npy" and item != "labels.npy":
            datas.append(item)


transform_test = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


accs = []
def robust_test(net):
    for file_index in datas:
        test_data = np.load("./CIFARC/"+file_index)
        test_label = np.load("./CIFARC/"+"labels.npy")
        test_data = np.transpose(test_data, [0, 3, 1, 2])
        test_data = torch.FloatTensor(test_data).cuda()
        test_label = torch.LongTensor(test_label).cuda()
        tensor_dataset = TensorDataset(test_data, test_label)
        tensor_loader = DataLoader(tensor_dataset, batch_size=2000)
        acc = evaluate(net, tensor_loader)
        accs.append(acc)
        print("Accuracy on "+file_index+" is %.4f" % acc)


def evaluate(net, tensor_loader):
    #   net.cpu()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for data in tensor_loader:
            net.eval()
            images, labels = data
            images, labels = images/255.0, labels.cuda()
            normalize_images = []
            images = images.cpu()
            for index in range(images.size(0)):
                normalize_images.append(torch.unsqueeze(transform_test(images[index]), dim=0))
            images = torch.cat(normalize_images, dim=0).cuda()
            outputs = net([images])
            outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += float(labels.size(0))
            correct += float((predicted == labels).sum())
        acc = (100 * correct / total)
    #   net.cuda()
    return acc
