import argparse
import numpy as np
import os
import random
import ujson

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, sampler
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from MPChallengeDataset import MPChallengeDataset
import custom_models
from trainer import CNNTrainer

parser = argparse.ArgumentParser(description='VAE Speech')

parser.add_argument('--dataset_dir', type=str, metavar='S',
                    help='dataset directory')
parser.add_argument('--cuda', type=bool, default=False, metavar='b',
                    help='use cuda if possible (default: False)')
parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='N',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--num-epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 60)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 24)')
parser.add_argument('--checkpoint-interval', type=int, default=1, metavar='N',
                    help='Interval between epochs to print loss and save model (default: 1)')
parser.add_argument('--model-name', type=str, default='CNN', metavar='S',
                    help='model name (for saving) (default: CNN)')
parser.add_argument('--mode', type=str, default='train', metavar='S',
                    help='(operation mode (default: train); to test single example: test')

def load_dataset(dataset_dir, train_subset=0.8):
    """
    load_dataset function loads the dataset and creates data loaders for
    both test and train data
    """

    # data augmentation on train set
    train_transfs = transforms.Compose([
        # transforms.RandomResizedCrop(size=100, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    ])

    test_transfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = MPChallengeDataset(dataset_dir, transform = train_transfs)
    test_dataset = MPChallengeDataset(dataset_dir, transform = test_transfs)

    indices = list(range(len(train_dataset)))
    random.shuffle(indices)

    split = int(np.floor(len(train_dataset) * train_subset))

    # create train / test split, load the same if created once
    if (os.path.isfile('train_split_indices.json') and os.path.isfile('test_split_indices.json')):
        with open('train_split_indices.json') as f:
            train_indices = ujson.load(f)
        with open('test_split_indices.json') as f:
            test_indices = ujson.load(f)
    else:
        with open('train_split_indices.json', 'w') as f:
            ujson.dump(indices[:split], f)
        with open('test_split_indices.json', 'w') as f:
            ujson.dump(indices[split:], f)

        train_indices = indices[:split]
        test_indices = indices[split:]

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    test_sampler = sampler.SubsetRandomSampler(test_indices)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler = train_sampler, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler = test_sampler, drop_last=False, **kwargs)

    return train_loader, test_loader

def train(args):
    """
    train function, trains the model
    """

    model = custom_models.TransferCNN()
    # print(model)
    # sys.exit()
    # model = models.vgg11_bn(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # model.classifier[6] = nn.Sequential(
    #     nn.Linear(4096, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(256, 4),
    #     nn.LogSoftmax(dim=1))

    train_loader, test_loader = load_dataset(args.dataset_dir)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = CNNTrainer(args, model, train_loader, test_loader, optimizer)

    print("Model Name: {}".format(args.model_name))
    print("Starting training...")

    trainer.train_epochs()

def test(args):
    """
    test function, tests the model by creating class Probabilities for a
    single sample chosen randomly from test set and calculating accuracy on
    the whole test set
    """

    model = custom_models.CNN()
    model.load_state_dict(torch.load('experiments/'+args.model_name, map_location=lambda storage, loc: storage))
    model.eval()

    train_loader, test_loader = load_dataset(args.dataset_dir)

    correct = 0
    total = 0

    with torch.no_grad():

        it = iter(test_loader)
        single_sample = next(it)
        print(single_sample[2][0])
        print("Testing single example: {}".format(single_sample[2][0]))
        output = torch.exp(model(single_sample[0][0].unsqueeze(0)))
        print("Class Probabilities")
        print(output)

        print("Evaluating Accuracy...")

        for data, label, _ in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))

if __name__ == '__main__':

    args = parser.parse_args()
    args.writer = SummaryWriter('experiments/exp_7')

    if args.dataset_dir is None :
        raise TypeError("Dataset directory not passed as argument (see -h)")

    if (args.cuda and torch.cuda.is_available()):
        args.use_cuda = True
    else:
        args.use_cuda = False

    print("Using CUDA: {}".format(args.use_cuda))

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        print("No --mode provided, options: train, test (see -h)")
