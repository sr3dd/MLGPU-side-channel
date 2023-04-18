import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from rnet import compute_accuracy

batch_size = 128
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

model = torch.load('Resnet18_mnist.pt')
model.eval()
model = model.to(device)
    
print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device)))
