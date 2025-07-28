import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import os
from PIL import Image
from sklearn.model_selection import train_test_split

#used to compare different architecture performances
Architectures = {
    'Small' : {
        'conv1_out' : 6,
        'conv2_out' : 16,
        'fc1_out' : 120,
        'fc2_out' : 84,
    },
    'Med' : {
        'conv1_out' : 16,
        'conv2_out' : 32,
        'fc1_out' : 256,
        'fc2_out' : 128,
    },
    'Large' : {
        'conv1_out' : 32,
        'conv2_out' : 64,
        'fc1_out' : 512,
        'fc2_out' : 256
    },
}

#defines model
class DrivingNet(nn.Module):
    def __init__(self, conv1_out, conv2_out, fc1_out, fc2_out):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_out, kernel_size = 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(conv1_out,conv2_out,kernel_size = 5)
        self.fc1 = nn.Linear(conv2_out*5*5, fc1_out)
        self.fc2 = nn.Linear(fc1_out,fc2_out)
        self.fc3 = nn.Linear(fc2_out,3)

    def forward(self,x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 


def createNetwork(learning_rate, arch_name, printToggle = False):
    #grabs the parameters depending on architecture
    sizes = Architectures[arch_name]
    #declares network for training
    net = DrivingNet(**sizes)
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    return net, lossfun, optimizer


def trainModel(batch_size, learning_rate, arch_name):

    #we define our training data and test data.  ]
    train_data, test_data, train_labels, test_labels = train_test_split(images,labels, test_size = .1)
    train_data = TensorDataset(train_data, train_labels)
    test_data = TensorDataset(test_data, test_labels)


    batchsize = batch_size
    train_loader = DataLoader(train_data, batch_size = batchsize, shuffle = True, drop_last = True)
    test_loader = DataLoader(test_data, batch_size = test_data.tensors[0].shape[0])

    numepochs = 150


    #calling model creation function
    net, lossfun, optimizer = createNetwork(learning_rate, arch_name)

    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc = []

    #go through our epochi and save data

    for epochi in range(numepochs):
        net.train()
        batchAcc = []
        batchLoss = []
        for X,y in train_loader:
            yHat = net(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchLoss.append(loss.item())
            
            preds = torch.argmax(yHat,dim = 1)
            correct = preds == y
            accuracy = correct.float().mean()
            batchAcc.append(accuracy.item())



        trainAcc.append(100*np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)

        net.eval()
        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)
        
        preds = torch.argmax(yHat, dim=1)
        correct = (preds == y).float()
        accuracy = 100 * correct.mean()
        testAcc.append(accuracy.item())

    
    return trainAcc,testAcc,losses,net


#read labels to check acc
df = pd.read_csv("labels.csv")

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5] *3)
])

images = []
labels = []
label_map = {
    'forward': 0,
    'left': 1,
    'right': 2,
}


#getting data
for index,row in df.iterrows():
    img_path = os.path.join('data', row[0])

    if not os.path.exists(img_path) or row[1] == 'stop':
        continue

    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image)
    labels.append(label_map[row[1]])

    images.append(image_tensor)

images = torch.stack(images)
labels = torch.tensor(labels)


print("Starting")

#defining parameters to train use when training the model
batch_size = 32
learning_rate = 0.001
arch_name = "Med"

#sending parameters through 
trainAcc, testAcc, losses, net = trainModel(batch_size, learning_rate, arch_name)


#plot data
plt.figure()
plt.plot(trainAcc, label='Train Accuracy')
plt.plot(testAcc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)

plt.show()


#old code used to determine best model architecture
# results = []

# for arch_name in Architectures:
#     for learning_rate in [0.01, 0.001, 0.0001]:
#         for batch_size in [16, 32, 64]:
#             trainAcc, testAcc, losses, net = trainModel(batch_size, learning_rate, arch_name)
#             results.append({

#                 'arch' : arch_name, 
#                 'batch_size' : batch_size,
#                 'learning_rate': learning_rate,
#                 'final_test_acc' : testAcc[-1],
#                 'best_test': max(testAcc),
#                 'trainAcc' : trainAcc,
#                 'testAcc' : testAcc,
#                 'losses' : losses.tolist()
#             })
#             print("Next")

# df_results = pd.DataFrame(results)

# with open("experiment_results.pkl", "wb") as f:
#     pickle.dump(results, f)

# summary_df = df_results.drop(columns=['trainAcc', 'testAcc', 'losses'])
# summary_df.to_csv("experiment_summary.csv", index=False)

# plt.figure(figsize = (12,6))
# plt.plot(df_results['final_test_acc'], marker = 'o')
# plt.xlabel("Exp number")
# plt.ylabel("Final Test Acc")
# plt.grid(True)
# plt.show()



        