import torch  
import torch.nn as nn
from torchvision import transforms 
import sys 
sys.path.append("..")
from data import test_set,train_set
from torch.utils.data import DataLoader

class CIFAR10CNN(nn.Module):
    def __init__(self,in_channels=3,base_filters=32,fc_hidden=256,num_classes=10):
        super().__init__()
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
        self.conv1=nn.Conv2d(in_channels,base_filters,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(base_filters)
        self.conv2=nn.Conv2d(base_filters,base_filters,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(base_filters)
        self.shortcut1=nn.Conv2d(in_channels,base_filters,kernel_size=1)
        self.conv3=nn.Conv2d(base_filters,base_filters*2,kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(base_filters*2)
        self.conv4=nn.Conv2d(base_filters*2,base_filters*2,kernel_size=3,padding=1)
        self.bn4=nn.BatchNorm2d(base_filters*2)
        self.shortcut2=nn.Conv2d(base_filters,base_filters*2,kernel_size=1)
        self.fc1=nn.Linear(base_filters*2*8*8,fc_hidden)
        self.fc2=nn.Linear(fc_hidden,num_classes)
        self.dropout=nn.Dropout(p=0.3)
    def forward(self,x):
        identity=self.shortcut1(x)
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x))+identity)
        x=self.pool(x)
        identity=self.shortcut2(x)
        x=self.relu(self.bn3(self.conv3(x)))
        x=self.relu(self.bn4(self.conv4(x))+identity)
        x=self.pool(x)
        x=x.view(x.shape[0],-1)
        x=self.dropout(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
    
train_loader=DataLoader(train_set,batch_size=64,shuffle=True,num_workers=0)
test_loader=DataLoader(test_set,batch_size=64,shuffle=False,num_workers=0)

num_epochs=40
device="cuda"
model=CIFAR10CNN().to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
criterion=nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.4)

 
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    correct_val=0
    correct_train=0

    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        _,preds=torch.max(outputs,dim=1)
        correct_train+=(preds==labels).sum().item()
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    avg_loss=running_loss/len(train_loader)
    train_accuracy=correct_train/len(train_loader.dataset)
    with open("log2.txt","a")as f:
        f.write(f'Epoch [{epoch +1}/{num_epochs}] Loss:{avg_loss :.4f} Train_Accuracy:{train_accuracy :.2f} ')
    print(f'Epoch [{epoch +1}/{num_epochs}] Loss:{avg_loss :.4f} Train_Accuracy:{train_accuracy :.2f}')   
    with torch.no_grad():
        model.eval()
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,preds=torch.max(outputs,dim=1)
            correct_val+=(preds==labels).sum().item()
        accuracy=correct_val/len(test_loader.dataset)
        print(f'Val_Accuracy : {accuracy :.2f}')
        with open("log2.txt","a")as f:
            f.write(f'Val_Accuracy : {accuracy :.2f}\n')
        scheduler.step(accuracy)
torch.save(model.state_dict(), 'cifar10_cnn_residuals2.pth')   
