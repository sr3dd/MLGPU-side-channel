from resnet.model import Resnet18Mnist
from resnet.train import train
from resnet.mnist_dataloader import mnist_dataloader

from torch import optim
from torch import nn

def transfer_learning(train_epochs:int=20, early_stop_epochs:int=3, save_file_name:str="MNIST_RESNET18.pt"):
    '''
    INFO
    ----
    From a RESNET18 net pretrained on Imagenet, we iterate n epochs to train the net in MNIST.
    It serves for obtaining valid weights. No return allowed.
    :param train_epochs: number of epochs to train
    :param early_stop_epochs: stop epochs if there is no loss improvement
    :param save_file_name: save weights as a *.pt file
    :return there is no return. Weights are generated in the wkdir under save_file_name denomination.
    '''

    # Initialise the Resnet18 model (pretrained)
    resnet18_mnist = Resnet18Mnist()

    # Prepare the model for transfer learning training
    resnet18_mnist.training_prep()

    # Retrieve the model for training
    resnet18 = resnet18_mnist.model()

    # Loss and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(resnet18.parameters())

    # Import train_loader and valid_loader
    train_loader, valid_loader = mnist_dataloader()

    # Train: weights are generated in the working directory
    model, history = train(model=resnet18,
                           criterion=loss_function,
                           optimizer=optimizer,
                           train_loader=train_loader,
                           valid_loader=valid_loader,
                           save_file_name=save_file_name,
                           max_epochs_stop=early_stop_epochs,
                           n_epochs=train_epochs,
                           print_every=2
                           )

    # For analyzing loss and accuracy curves
    history.to_csv("MNIST_RESNET18.csv", sep=",", index=False)


if __name__ == '__main__':
    transfer_learning()
