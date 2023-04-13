from torchvision import models
import torch.nn as nn
import torch

class Resnet18Mnist:

    def __init__(self):

        self.resnet18 = models.resnet18(num_classes=10)

    def freeze_layers(self):
        # Freeze ResNet18 model weights (all layers)
        for param in self.resnet18.parameters():
            param.requires_grad = True

    def add_classifier(self):
        # change input layer needs to accept single channel instead of 3 
        # (MNIST images are single-channel = grayscale, whereas 
        # ImageNet are 3-channels = RGB).
        pass
        #self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def training_prep(self):
        # Prepare the net for transfer learning classifier-only training
        self.freeze_layers()
        self.add_classifier()

    def model(self) -> models.vgg16_bn:
        # Return the model only
        return self.resnet18

    def set_test_mode(self, weight_path: str):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.training_prep()
        self.resnet18.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

        if torch.cuda.is_available():
            self.resnet18.to("cuda")

    def evaluate(self, input):
        return self.resnet18(input)
        