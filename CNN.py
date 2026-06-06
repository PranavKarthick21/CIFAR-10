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
        self.conv1=nn.Conv2d(in_channels,base_filters,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(base_filters)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(base_filters,base_filters*2,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(base_filters*2)
        self.conv3=nn.Conv2d(base_filters*2,base_filters*4,kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(base_filters*4)
        self.fc1=nn.Linear(base_filters*4*4*4,fc_hidden)
        self.fc2=nn.Linear(fc_hidden,num_classes)
        self.dropout=nn.Dropout(p=0.5)
    def forward(self,x):
        x=self.pool(self.relu(self.bn1(self.conv1(x))))
        x=self.pool(self.relu(self.bn2(self.conv2(x))))
        x=self.pool(self.relu(self.bn3(self.conv3(x))))
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x
    
train_loader=DataLoader(train_set,batch_size=64,shuffle=True,num_workers=0)
test_loader=DataLoader(test_set,batch_size=64,shuffle=False,num_workers=0)

num_epochs=40
device="cuda"
model=CIFAR10CNN().to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
criterion=nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))

 
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    correct=0

    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss+=loss.item()
    avg_loss=running_loss/len(train_loader)
    print(f'Epoch [{epoch +1}/{num_epochs}] Loss:{avg_loss :4f}')   
    with torch.no_grad():
        model.eval()
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,preds=torch.max(outputs,dim=1)
            correct+=(preds==labels).sum().item()
        accuracy=correct/len(test_loader.dataset)
        print(f'Accuracy : {accuracy}')
torch.save(model.state_dict(), 'cifar10_cnn_residuals.pth')   
