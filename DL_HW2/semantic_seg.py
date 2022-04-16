import torch.optim as optim
from tqdm import tqdm
from network import ResNet18FCN, UNet
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset
import glob
import cv2
from sklearn.metrics import jaccard_score
from sklearn.utils.multiclass import type_of_target

class SegmentationDataset(Dataset):
    def __init__(self, train=True):
        dataset_path = "./data/seg_data/" + ("train" if train else "test")
        self.images = sorted(glob.glob(dataset_path+"/*/*/CameraRGB/*.png"))
        self.masks = sorted(glob.glob(dataset_path+"/*/*/CameraSeg/*.png"))
        self.resize_shape = (320, 416)

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)[:,:,2]

        channels=3
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.array(image).reshape((image.shape[0], image.shape[1], channels)).astype(np.float32) / 255.0
        mask = np.array(mask).reshape((image.shape[0], image.shape[1], 1))

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.transform_image(self.images[idx], self.masks[idx])
        sample = {'image': image, "mask": mask, 'idx': idx}

        return sample

trainset = SegmentationDataset(train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = SegmentationDataset(train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

def train_seg_model():
    net = UNet()
    net.cuda()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochs = 2
    for epoch in range(epochs):  # loop over the dataset multiple times
        with tqdm(total=len(trainset), desc =str(epoch)+"/"+str(epochs), miniters=int(50),unit='img') as prog_bar:
          for i, data in enumerate(trainloader, 0):
              # get the inputs; data is a list of [inputs, labels]
              inputs = data["image"]
              labels = data["mask"]
              inputs = inputs.cuda()
              labels = labels.cuda()

              # zero the parameter gradients
              optimizer.zero_grad()

              # forward + backward + optimize
              outputs = net(inputs)
              loss = criterion(outputs, labels[:,0,:,:].long())
              loss.backward()
              optimizer.step()

              prog_bar.set_postfix(**{'loss': np.round(loss.data.cpu().detach().numpy(),5)})
              prog_bar.update(4)

    return net

def test_seg_model(net, curr_data_loader, val_test="val"):
    num_images = len(curr_data_loader.dataset)
    net.eval()
    lossj = 0
    with torch.no_grad():
        for i,data in enumerate(testloader, 0):
            images = data["image"]
            labels = data["mask"]
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            x = labels.detach().cpu().numpy()
            y = predicted.detach().cpu().numpy()
            x = x.ravel()
            y = y.ravel()

            # Calculate pixel difference
            lossj += jaccard_score(x, y, average='macro')

    # Average the pixel difference
    print("Mean IU accuracy: ", 1 - lossj/num_images)

'''
MAIN FUNCTION
'''
if __name__ == '__main__':
    net = train_seg_model()
    test_seg_model(net, testloader)
