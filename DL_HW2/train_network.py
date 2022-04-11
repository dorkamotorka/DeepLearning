import torch.optim as optim
from tqdm import tqdm
from network import ResNet18Plain
from torch import nn
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
# Transforms can also be used for image augmentation - https://pytorch.org/vision/stable/transforms.html
transform = transforms.Compose([transforms.ToTensor()])

transform_test = transforms.Compose([transforms.ToTensor()])


trainset = torchvision.datasets.ImageFolder(root='./bird_data/train/',transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='./bird_data/test/', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

valset = torchvision.datasets.ImageFolder(root='./bird_data/valid/',transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

def test_bird_model(net, curr_data_loader, val_test="val"):
    # Testing

    criterion = nn.CrossEntropyLoss()

    num_images = len(curr_data_loader.dataset)
    gt_array = np.zeros(num_images)
    pred_array = np.zeros(num_images)

    correct = 0
    total = 0
    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(curr_data_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            gt_array[i*labels.size(0):(i+1)*labels.size(0)] = labels.detach().cpu().numpy()
            pred_array[i*labels.size(0):(i+1)*labels.size(0)] = predicted.detach().cpu().numpy()

    print('Accuracy of the network on %s images: %d %%' % (val_test, 100 * correct / total))
    print("Test loss: " + str(running_loss/(total/4)))
    return gt_array, pred_array

def train_bird_model(epochs, lr, wd):
    # Use the ResNet18 when implemented
    #net = ResNet18()
    net = ResNet18Plain()
    net.cuda()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        with tqdm(total=len(trainset), desc ='Epoch: '+str(epoch)+"/"+str(epochs), unit='img') as prog_bar:
            for i, data in enumerate(trainloader, 0):
                # Get the inputs; Data is a tuple of (images, labels)
                inputs, labels = data
                # Transfer the images and labels to the GPU.
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Clear the saved gradients of the previous iteration
                optimizer.zero_grad()

                outputs = net(inputs)
                # Calculate the loss value
                loss = criterion(outputs, labels)
                # Calculate the gradients using backpropagation
                loss.backward()
                # Update the weights of the network using the chosen optimizer
                optimizer.step()

                prog_bar.set_postfix(**{'loss': loss.data.cpu().detach().numpy()})
                prog_bar.update(batch_size)
        
    test_bird_model(net, testloader)

    return net

if __name__ == '__main__':
    #net = train_bird_model(10, 0.000001, 0) # 13
    #net = train_bird_model(10, 0.00001, 0) # 62
    #net = train_bird_model(10, 0.0001, 0) # 84
    #net = train_bird_model(10, 0.001, 0) # 73
    #net = train_bird_model(10, 0.01, 0) # 73
    #net = train_bird_model(10, 0.001, 0.0001) # 69 
    #net = train_bird_model(10, 0.0001, 0.0001) # 84
    #net = train_bird_model(10, 0.0001, 0.001) # 85 
    #net = train_bird_model(10, 0.0001, 0.01) # 78
    #net = train_bird_model(10, 0.00015, 0.00001) # 86
    #net = train_bird_model(10, 0.005, 0.00001) # 
    #net = train_bird_model(10, 0.005, 0.000015) # 
    #net = train_bird_model(10, 0.0001, 0.000015) # 86 
    #net = train_bird_model(10, 0.00015, 0.000015) # 87 BEST
    #net = train_bird_model(10, 0.0001, 0.000001) # 
    #net = train_bird_model(10, 0.00015, 0.00002) # 85 
    #net = train_bird_model(10, 0.00015, 0.000012) #  
    #net = train_bird_model(10, 0.00015, 0.000018) #  
    #net = train_bird_model(10, 0.00015, 0.00005) #  
    #net = train_bird_model(10, 0.0002, 0.000015) #
    #net = train_bird_model(10, 0.0003, 0.000015) #
    #net = train_bird_model(10, 0.0004, 0.000015) #
    #net = train_bird_model(20, 0.00015, 0.000015) # 88 (batch_size=64)
    net = train_bird_model(20, 0.00015, 0.000015) # 

    # probaj batch size 32 z 85 in 84mi
    # Povečaj število epoch

    gt_array, pred_array = test_bird_model(net, testloader)

    # Displaying random misclassified images
    fig = plt.figure(figsize=(24, 24))
    columns = 2
    rows = 6
    mistake_indices = np.nonzero(gt_array != pred_array)[0]
    for i in range(rows):
        chosen_index = mistake_indices[np.random.randint(len(mistake_indices))]
        input, label = testset[chosen_index]
        pred_label = pred_array[chosen_index].astype(np.int32)
        pred_cls_samples = np.nonzero(gt_array == pred_label)[0]
        pred_cls_input, _ = testset[pred_cls_samples[np.random.randint(len(pred_cls_samples))]]

        img = input.detach().numpy().transpose((1,2,0))
        ax = fig.add_subplot(rows, columns, columns*i+1)
        im_title = "GT: "+trainset.classes[label] + " P: "+trainset.classes[pred_label]
        ax.set_title(im_title, fontstyle='italic')
        plt.imshow(img)

        img_sample = pred_cls_input.detach().numpy().transpose((1,2,0))
        ax = fig.add_subplot(rows, columns, columns*i+2)
        im_title = "Pred. Cls. Example: "+trainset.classes[pred_label]
        ax.set_title(im_title, fontstyle='italic')
        plt.imshow(img_sample)

    plt.show()
