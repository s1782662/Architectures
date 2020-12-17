from torchvision import transforms
from torch.utils.data import DataLoader
from fMNIST.dataloaders import FashionMNIST
from fMNIST.cnn.networks import LeNet
from fMNIST.cnn.networks import LeNetEnsemble
from fMNIST.utils import constants
from torchvision.utils import make_grid
import torch.nn.functional as F 
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import matplotlib
import torch 

def accuracy(pred_labels, labels):
    pred_labels = torch.argmax(pred_labels,dim=1)
    correct = pred_labels.eq(labels)
    return torch.mean(correct.float())

def plot_accuracy(lloss,laccuracy):
    fig, axs = plt.subplots()
    ax2 = axs.twinx()
    axs.plot(lloss,label='loss',color='g')
    ax2.plot(laccuracy, label='accuracy',color='b')
    axs.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    plt.title('One LeNet training loss and accuracy')
    plt.show()


def main():
             
    # set the transforms
    trTransforms = transforms.Compose([
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()
                    ])

    teTransforms = transforms.Compose([
                       transforms.ToTensor()
                    ])


    # dataset
    train = FashionMNIST(
                constants.DIR, 
                constants.TRAIN_FILE,
                transform=trTransforms
                )

    test = FashionMNIST(
                constants.DIR, 
                constants.TEST_FILE,
                transform=teTransforms
                )

    # number of Classes
    nClasses = len(set(train.labels))


    #setup the dataloader
    trLoader = DataLoader(train,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True,
                    num_workers=constants.NWORKERS
                    )

    teLoader = DataLoader(test,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=False,
                    num_workers=constants.NWORKERS
                    )

    # testing dataloaders 
    samples, labels = iter(trLoader).next()       
    nClasses = 10

    #model = LeNet(nClasses)
    model = LeNetEnsemble(nClasses)
    # move the model to gpu mode
    model.to(constants.DEVICE)

    # setup the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # schedule the learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000,
        10000, 15000], gamma=0.5)
    
    criterion = nn.CrossEntropyLoss()
    lloss,laccuracy = [],[]
    total_loss,total_accuracy=0,0
    # train the system
    model.train()
    itr = 1
    for epoch in range(constants.EPOCHS):
        
        for batch_idx,(features, labels) in enumerate(trLoader):
            
            features = features.to(constants.DEVICE)
            labels   = labels.to(constants.DEVICE)
            
            # nullify the gradients
            optimizer.zero_grad()

            # compute the output from the model
            pred_labels = model(features)

            # calculate the cross entropy loss
            loss = criterion(pred_labels, labels)

            # compute the backward gradients
            loss.backward()                
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy(pred_labels, labels)
            
            # 
            scheduler.step()
            
            if itr%constants.ITERATIONS == 0:
                print('Epoch: %03d/%3d | Iterations:%04d | Cost: \
                        %.4f | Accuracy: \
                        %.4f'%(epoch+1,constants.EPOCHS,itr,total_loss/constants.ITERATIONS,total_accuracy/constants.ITERATIONS))

                lloss.append(total_loss/constants.ITERATIONS)
                laccuracy.append(total_accuracy/constants.ITERATIONS)
                total_loss,total_accuracy = 0,0
            itr += 1
    
    plot_accuracy(lloss,laccuracy)              
    model.eval()
    test_accuracy = 0.0

    for features,target in teLoader:
        with torch.no_grad():
            features, labels = features.to(constants.DEVICE),target.to(constants.DEVICE)
            logits = model(features)
            test_accuracy += accuracy(logits,labels)

    print('Test accuracy of the model is {}'.format(round((test_accuracy.item()/len(teLoader)) * 100.0,2)))
    



if __name__ == '__main__':
    main()
    

                    



