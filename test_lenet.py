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


test_dataset = datasets.MNIST(root='files/test', 
                              train=False, 
                              transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

model = torch.load('lenet_mnist.pt')
model.eval()
model = model.to(device)

all_correct_num = 0
all_sample_num = 0
    
for idx, (test_x, test_label) in enumerate(test_loader):
    test_x = test_x.to(device)
    test_label = test_label.to(device)
    predict_y = model(test_x.float()).detach()
    predict_y =torch.argmax(predict_y, dim=-1)
    current_correct_num = predict_y == test_label
    all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
    all_sample_num += current_correct_num.shape[0]
acc = all_correct_num / all_sample_num
print('accuracy: {:.3f}'.format(acc), flush=True)
