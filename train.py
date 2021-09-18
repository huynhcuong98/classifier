from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchsummary import summary
import time
# from face_classify.dataloader import Image_Loader
from dataloader import Image_Loader
from model import Network, Arcface
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import os
import shutil

def plot_result(epoch, loss_train, loss_test, accuracy,  weight_path):
    #plot 1:
    plt.subplot(2, 1, 1)
    plt.plot(epoch, loss_train )
    plt.plot(epoch, loss_test )
    plt.legend(["train_loss", "val_loss"])
    
    plt.ylabel("loss")


    #plot 2:
    plt.subplot(2, 1, 2)
    plt.plot(epoch, accuracy )
    plt.legend(["accuracy"])
    plt.xlabel("epochs")
    
    plt.savefig(weight_path.replace('.pt','.jpg'))
    plt.close()


def train(args, model, device, train_loader, optimizer, epoch, CE, weight_path, class_names):
    model.train()
    
    # Initialize the prediction and label lists(tensors)
    nb_classes = len(class_names)
    print('nb_classes:', nb_classes)
    confusion_matrix = np.zeros((nb_classes, nb_classes))

    tr_loss = 1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = CE(output, target)
        tr_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

        # Append batch prediction results
        pred = output.argmax(dim=1, keepdim=True)
        # print('pred, target:', torch.cat((pred,target.view(-1,1)), 1))
        for t, p in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1        
    # get average train loss
    tr_loss/= len(train_loader)

    # df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    # heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    # heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=10)
    # heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=10)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.savefig(weight_path.replace('.pt','_CF_Train.jpg'))
    # plt.close()

    print('\ntrain set: Average loss: {:.4f}\n'.format(tr_loss ))

    return tr_loss

def test(model, device, test_loader, weight_path, CE, class_names):
    global min_loss
    model.eval()
    test_loss = 0
    correct = 0

    # Initialize the prediction and label lists(tensors)
    nb_classes = len(class_names)
    confusion_matrix = np.zeros((nb_classes, nb_classes))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            t1 = time.time()
            output = model(data)
            # print(output)
            fps = int(1/(time.time()-t1))
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += CE(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Append batch prediction results
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    
    

    test_loss /= len(test_loader.dataset)
    if test_loss < min_loss:
      min_loss = test_loss
      torch.save(model.state_dict(), weight_path)
      print("save model at: ", min_loss)

      df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
      heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
      heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=10)
      heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=10)
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      plt.savefig(weight_path.replace('.pt','_CF_Test.jpg'))
      plt.close()

    print('\nTest set: FPS: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(fps,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



    return test_loss, (100. * correct / len(test_loader.dataset))
def main():
    global train_loss, test_loss, ls_epochs, ls_accuracy
    # Default value
    batch_size = 16

    # Load data for training
    img_size =[64,64]
    train_data = Image_Loader(root_path='./train.csv',image_size= img_size, transforms_data=True)
    # Load data for testing
    test_data = Image_Loader(root_path='./test.csv',image_size= img_size,   transforms_data=True)

    lb = pd.read_csv('label.csv', header = None)
    class_names = lb.values[:,0]
    weight_path = "./weights/cnn_res2.pt"

    total_train_data = len(train_data)
    total_test_data = len(test_data)

    print("test_data: ", total_test_data)
    print("train_data: ", total_train_data)

    # Generate the batch in each iteration for training and testing
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    model = Network(len(class_names)).to(device)
    print(model)
    summary(model, (3, 64, 64)) #summary(your_model, input_size=(channels, H, W))

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    CE = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    # scheduler = MultiStepLR(optimizer, milestones=[10,30], gamma=0.1)
    for epoch in range(1, args.epochs + 1):
        train_loss.append(train(args, model, device, train_loader, optimizer, epoch, CE, weight_path, class_names)) 
        loss_t, accuracy = test(model, device, test_loader, weight_path, CE, class_names)
        test_loss.append(loss_t)
        ls_accuracy.append(accuracy)
        ls_epochs.append(epoch)
        scheduler.step()
        plot_result(ls_epochs, train_loss, test_loss, ls_accuracy, weight_path)

    # if args.save_model:
    #     torch.save(model.state_dict(), "cnn_3cls.pt")

def create_fd():
  fd =  'weights'
  if os.path.exists(fd):
      shutil.rmtree(fd)
  os.makedirs(fd)
  
if __name__ == '__main__':
    # global min_loss
    min_loss = 10
    train_loss = [] 
    test_loss = []
    ls_epochs = []
    ls_accuracy = []
    create_fd()
    main()