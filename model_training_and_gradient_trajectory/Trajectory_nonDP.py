DEVICE='cuda:1'
DIMENSION=224
MINI_BS=100
BS=1000
LR=1e-4
MODEL='vit_base_patch16_224'
CLIPPING_MODE='nonDP'
#MODEL_PATH='./checkpoints/nonDP_model_epoch_2.pth'
MODEL_PATH='./checkpoints/nonDP_model_epoch_2.pth'


n_acc_steps = BS // MINI_BS # gradient accumulation steps

import torch

device= torch.device(DEVICE if torch.cuda.is_available() else "cpu") #默认为cuda:0
print("device:",device)

print('==> Preparing data..')

import torchvision
transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize(DIMENSION),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transformation)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=MINI_BS, shuffle=True, num_workers=4) # shuffle会在数据加载前打乱数据集

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

import timm
from opacus.validators import ModuleValidator
import torch.nn as nn
import torch.optim as optim

checkpoint = torch.load(MODEL_PATH)

num_classes=100
print('==> Building model..', MODEL,'; BatchNorm is replaced by GroupNorm. Mode: ', CLIPPING_MODE)
net = timm.create_model(MODEL,pretrained=True,num_classes=num_classes)
net = ModuleValidator.fix(net); # fix使其能用于DP训练
net.load_state_dict(checkpoint['model_state_dict'])
net=net.to(device) 

print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=LR)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


#先进行训练，存储模型并得到true_gradient

from tqdm import tqdm
import numpy as np
SHOW_STEPS=100

STAGE='mid'
NUM=0
model_path = f'./nonDP_checkpoints/{STAGE}_{NUM}.pth'
torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_path)
print(f'Model saved to {model_path}')


def get_gradient(net):
    current_gradient = []
    for param in net.parameters():
        current_gradient.append(param.grad.view(-1).detach().cpu().numpy())
    # print(len(current_gradient))
    # print([i.size for i in current_gradient])
    gradient=np.concatenate(current_gradient)

    return gradient

true_gradient=[]

net.train()
train_loss = 0
correct = 0
total = 0


for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)): #这里的batch_idx应该相当于step？
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
    loss = criterion(outputs, targets) # 交叉熵函数作为LossFunction
    loss.backward()
    # 每个mini_batch都有自己的一个true gradient
    if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
        present_true_gradient=get_gradient(net)
        true_gradient.append(present_true_gradient)
        print(len(true_gradient))
        optimizer.step() # 每积累n_acc_steps步的梯度后进行一次更新参数(每执行logical batch后更新一次)
        optimizer.zero_grad()

        #保存每个batch_size训练后的模型
        NUM=(batch_idx + 1) // n_acc_steps
        model_path = f'./nonDP_checkpoints/{STAGE}_{NUM}.pth'
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        print(f'Model saved to {model_path}')
        
        
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()



    # print log
    if ((batch_idx + 1) % SHOW_STEPS == 0) or ((batch_idx + 1) == len(trainloader)):
        #privacy_spent = privacy_engine.get_privacy_spent(accounting_mode="all", lenient=False)
        tqdm.write("----------------------------------------------------------------------------------------")
        tqdm.write('Epoch: {}, step: {}, Train Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
            'none', batch_idx + 1, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        #tqdm.write("Privacy Cost: ε_rdp: {:.3f} | α_rdp: {:.1f} | ε_low: {:.3f} | ε_estimate: {:.3f} | ε_upper: {:.3f}".format(
            #privacy_spent["eps_rdp"], privacy_spent["alpha_rdp"], privacy_spent["eps_low"], privacy_spent["eps_estimate"], privacy_spent["eps_upper"]))

np.save(f'./gradients/nonDP_{STAGE}_true_gradients.npy', true_gradient)


print('Epoch: ', 'none', "total: ", len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# 对每个true gradient分别计算相应的per-sample gradient
MINI_BS=1
BS=1
n_acc_steps = BS // MINI_BS # gradient accumulation steps

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=MINI_BS, shuffle=True, num_workers=4) # shuffle会在数据加载前打乱数据集

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

import timm
from opacus.validators import ModuleValidator
import torch.nn as nn
import torch.optim as optim

total_per_sample_gradient=[]

for i in range(50):
    print('i=',i)
    MODEL_PATH='nonDP_checkpoints/'+STAGE+"_"+str(i)+'.pth'
    checkpoint = torch.load(MODEL_PATH)
    num_classes=100
    print('==> Building model..', MODEL,'; BatchNorm is replaced by GroupNorm. Mode: ', CLIPPING_MODE)
    net = timm.create_model(MODEL,pretrained=True,num_classes=num_classes)
    net = ModuleValidator.fix(net); # fix使其能用于DP训练
    net.load_state_dict(checkpoint['model_state_dict'])
    net=net.to(device) 

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    i_per_sample_gradient=[]

    net.train()
    train_loss = 0
    correct = 0
    total = 0


    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets) # 交叉熵函数作为LossFunction
        loss.backward()

        if((batch_idx+1)%50==0):
            present_per_sample_gradient=get_gradient(net)
            i_per_sample_gradient.append(present_per_sample_gradient)
        
        optimizer.step() # 每积累n_acc_steps步的梯度后进行一次更新参数(每执行logical batch后更新一次)
        optimizer.zero_grad()


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        if(batch_idx+1==1000):
            break
    print(len(i_per_sample_gradient))
    np.save(f'./gradients/nonDP_{STAGE}_{i}_gradients.npy', i_per_sample_gradient)
    #total_per_sample_gradient.append(i_per_sample_gradient)



