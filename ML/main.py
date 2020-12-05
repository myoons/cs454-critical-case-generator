import torch
from utils.dataloader import train_loader, val_loader

import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch.nn as nn
import torch.optim as optim

from models.small import smallNet
from models.medium import mediumNet
from models.res import Res18

from utils.seed import set_seed
from utils.lr import adjust_learning_rate

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Pytorch CS454 Image Classification Critical Case Generator')
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')


# Our Dataset Classes
classes = ('airplane', 'cat', 'dog', 'motorbike', 'person', 'dog')

def train(model, device, train_loader, val_loader, optimizer, criterion, epochs):

    print("Start Training \t Epochs : {} \t".format(epochs, model))
    
    model.train()
    
    best_accuracy = 0.5

    for epoch in range(epochs) : # Iterate Learning

        for batch_idx, data in enumerate(train_loader) :

            inputs, labels = data[0].to(device), data[1].to(device) # Batchsize : 30
            optimizer.zero_grad()
            
            # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
        (correct, total, class_correct, class_total) = eval(model, device)

        print("\t Epoch {}  Done \t Loss = {} \t Accuracy = {}".format(epoch, loss, (correct/total)))

        if (correct / total) > best_accuracy:
            best_accuracy = correct / total

            print("Saving the Best Model")
            parsing(correct, total, class_correct, class_total)
            PATH = './{}.pth'.format(args.model)
            torch.save(model.state_dict(), PATH)
        
        adjust_learning_rate(args, optimizer, epoch)


    print('Finished')

    return model

def eval(model, device):

    model.eval()

    print("Start Evaluating")

    correct = 0
    total = 0

    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))

    # Testing 하는 과정이므로 grad 추적 불필요
    with torch.no_grad() :
    
        for data in val_loader :
            
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            correct_images = (predicted == labels).squeeze()

            for i in range(inputs.size(0)) :
                label = labels[i]
                class_correct[label] += correct_images[i].item()
                class_total[label] += 1
                
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct, total, class_correct, class_total

def parsing(correct, total, class_correct, class_total):

    print('Accuracy of the network on the 50 test images : %d %%' % (100 * correct / total))
    print('----------------------------------------------------------------------------------')

    for i in range(5) :
        print('Accuracy of %4s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def main():

    # Is GPU Available?
    #device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    print("Current Using device : {}".format(device))

    if args.model == 'small' :
        model = smallNet()
    elif args.model == 'medium' :
        model = mediumNet()
    elif args.model == 'res' :
        model = Res18()
    else :
        print('model argument should be small or vgg')

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    trained_model = train(model, device, train_loader, val_loader, optimizer, criterion, args.epochs)

if __name__ == "__main__":
    global args
    args = parser.parse_args()
    set_seed(args)
    main()









