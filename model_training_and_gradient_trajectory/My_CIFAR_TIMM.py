'''Train CIFAR10/CIFAR100 with PyTorch.'''
def main(args):
    if (args.clipping_mode=='nonDP'):
        clipping_mode="nonDP"
    if args.clipping_mode not in ['nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT']:
        print("Mode must be one of 'nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT'")
        return None

    device= torch.device(args.device if torch.cuda.is_available() else "cpu") #默认为cuda:0
    print("device:",device)
    # Data
    print('==> Preparing data..')

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.dimension),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])


    if args.cifar_data=='CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transformation)
        num_classes=10
    elif args.cifar_data=='CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transformation)
        num_classes=100
    elif args.cifar_data=='SVHN':
        trainset = torchvision.datasets.SVHN(root='data/', split='train', download=True, transform=transformation)
        testset = torchvision.datasets.SVHN(root='data/', split='test', download=True, transform=transformation)
        num_classes=10
    else:
        return "Must specify datasets as CIFAR10 or CIFAR100 or SVHN"
         
 
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4) # shuffle会在数据加载前打乱数据集

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    n_acc_steps = args.bs // args.mini_bs # gradient accumulation steps

    # Model
    print('==> Building model..', args.model,'; BatchNorm is replaced by GroupNorm. Mode: ', args.clipping_mode)
    net = timm.create_model(args.model,pretrained=True,num_classes=num_classes)
    net = ModuleValidator.fix(net); # fix使其能用于DP训练
    net=net.to(device) 

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if 'BiTFiT' in args.clipping_mode: # not needed for DP-BiTFiT but use here for safety
        for name,param in net.named_parameters():
            if '.bias' not in name: # BiTFiT只对偏置项进行微调
                param.requires_grad_(False)

    # Privacy engine
    if 'nonDP' not in args.clipping_mode:
        sigma=get_noise_multiplier( # Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta) at the end of epochs, with a given sample_rate
                target_epsilon = args.epsilon,
                target_delta = args.delta,
                sample_rate = args.bs/len(trainset),
                epochs = args.epochs,
            )

        if 'BK' in args.clipping_mode:
            clipping_mode=args.clipping_mode[3:]
        else:
            clipping_mode='ghost'

        if args.clipping_style in [['all-layer'],['layer-wise'],['param-wise']]:
            args.clipping_style=args.clipping_style[0]
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(trainset),
            noise_multiplier=sigma,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
            clipping_style=args.clipping_style,
            origin_params=args.origin_params,#['patch_embed.proj.bias'],
        )
        privacy_engine.attach(optimizer)
    else:
        privacy_engine=None


    # the json file path is generated automatically according to the args.
    # The EXPERIMENT_NAME consisit of dataset/clippingMode/Time should be enough, detailed args stored in another file
    TIME=datetime.now().strftime("%m_%d_%H_%M_%S")
    if args.exp_name is None:
        EXPERIMENT_NAME=TIME+"_"+str(args.cifar_data)+"_"+str(args.clipping_mode)
    else:
        EXPERIMENT_NAME=args.exp_name+"_at_"+TIME
    FILE_PATH="./result/"+EXPERIMENT_NAME+"/"
    print("JSON files are saved in: ",FILE_PATH)

    def save_json(tmp_data:dict,file_path:str):
        # load exist data
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data = []

        # write data back
        data.append(tmp_data)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)


    # save all the params
    experiment_params={
        "experiment_name":"PyTorch CIFAR DP Training",
        "note":("None" if args.note==None else args.note),
        "experiment_time":TIME,
        "dataset":args.cifar_data,
        "trainset_size":len(trainset),
        "testset_size":len(testset),
        "model":args.model,
        "dimension":args.dimension,
        "device":(args.device if torch.cuda.is_available() else "cpu"),
        "optimizer":optimizer.__class__.__name__,
        "lr":args.lr,
        "num_epochs":args.epochs,
        "mini_bs":args.mini_bs,#physical
        "bs":args.bs,#logical
        "target_epsilon":args.epsilon if 'nonDP' not in args.clipping_mode else 'nonDP',
        "target_delta":args.delta if 'nonDP' not in args.clipping_mode else 'nonDP',
        "max_grad_norm":args.max_grad_norm if 'nonDP' not in args.clipping_mode else 'nonDP',
        "clipping_mode":args.clipping_mode,
        "actual_clipping_mode":clipping_mode if 'nonDP' not in args.clipping_mode else 'nonDP',
        "clipping_style":args.clipping_style if 'nonDP' not in args.clipping_mode else 'nonDP',
        "noise_multiplier":sigma if 'nonDP' not in args.clipping_mode else 'nonDP',
        "origin_params":args.origin_params,#['patch_embed.proj.bias'],
        "save_log":args.save_log,
    }
    save_json(experiment_params,FILE_PATH+"params.json")



    SHOW_STEPS=25 # 每多少步进行一次展示打印
    SAVE_STEPS=25 # 每进行多少步进行一次保存数据

    def train(epoch,privacy_engine):

        net.train()
        train_loss = 0
        correct = 0
        total = 0

   
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)): #这里的batch_idx应该相当于step？
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets) # 交叉熵函数作为LossFunction

            # if("resnet" in args.model):
            #     loss.requires_grad = True
            #     #print("using resnet, set loss.requires_grad = True")
            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step() # 每积累n_acc_steps步的梯度后进行一次更新参数(每执行logical batch后更新一次)
                # moniter insert here (peak) - min/max
                optimizer.zero_grad()
                # moniter insert here (reset)
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # save json logs
            if ((((batch_idx + 1) % SAVE_STEPS == 0) or ((batch_idx + 1) == len(trainloader))) and args.save_log!=0):
                if 'nonDP' not in clipping_mode:
                    privacy_spent = privacy_engine.get_privacy_spent(accounting_mode="all", lenient=False)
                else:
                    privacy_spent=None
                eval_results=get_eval_results()
                data={#后续会改
                    "epoch":epoch,
                    "step":batch_idx+1,
                    "train_loss":train_loss / (batch_idx + 1),
                    "train_acc":correct / total,
                    "eval_loss":eval_results["eval_loss"],
                    "eval_acc":eval_results["eval_acc"],
                    "epsilon_rdp":privacy_spent["eps_rdp"] if 'nonDP' not in args.clipping_mode else 'nonDP',
                    "alpha_rdp":privacy_spent["alpha_rdp"] if 'nonDP' not in args.clipping_mode else 'nonDP',
                    "eps_low":privacy_spent["eps_low"] if 'nonDP' not in args.clipping_mode else 'nonDP',
                    "epsilon_estimate":privacy_spent["eps_estimate"] if 'nonDP' not in args.clipping_mode else 'nonDP',
                    "eps_upper":privacy_spent["eps_upper"] if 'nonDP' not in args.clipping_mode else 'nonDP',
                }
                save_json(data,FILE_PATH+"train_log.json")

            # print log
            if ((batch_idx + 1) % SHOW_STEPS == 0) or ((batch_idx + 1) == len(trainloader)):
                #privacy_spent = privacy_engine.get_privacy_spent(accounting_mode="all", lenient=False)
                tqdm.write("----------------------------------------------------------------------------------------")
                tqdm.write('Epoch: {}, step: {}, Train Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
                    epoch, batch_idx + 1, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                #tqdm.write("Privacy Cost: ε_rdp: {:.3f} | α_rdp: {:.1f} | ε_low: {:.3f} | ε_estimate: {:.3f} | ε_upper: {:.3f}".format(
                    #privacy_spent["eps_rdp"], privacy_spent["alpha_rdp"], privacy_spent["eps_low"], privacy_spent["eps_estimate"], privacy_spent["eps_upper"]))


        print('Epoch: ', epoch, "total: ", len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if 'nonDP' not in clipping_mode:
            privacy_spent = privacy_engine.get_privacy_spent(accounting_mode="all", lenient=False)
            print("Privacy Cost: ",privacy_spent)

        # save final results of training 
        train_final_result={
            "mode":"train",
            "epoch:":epoch,
            "train_loss":train_loss/(batch_idx+1),
            "train_acc":correct/total,
            "eps_rdp":privacy_spent["eps_rdp"] if 'nonDP' not in args.clipping_mode else 'nonDP',
            "alpha_rdp":privacy_spent["alpha_rdp"] if 'nonDP' not in args.clipping_mode else 'nonDP',
            "eps_low":privacy_spent["eps_low"] if 'nonDP' not in args.clipping_mode else 'nonDP',
            "eps_estimate":privacy_spent["eps_estimate"] if 'nonDP' not in args.clipping_mode else 'nonDP',
            "eps_upper":privacy_spent["eps_upper"] if 'nonDP' not in args.clipping_mode else 'nonDP',
        }
        save_json(train_final_result,FILE_PATH+"final_results.json")

    def test(epoch,privacy_engine):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            if 'nonDP' not in clipping_mode:
                privacy_spent = privacy_engine.get_privacy_spent(accounting_mode="all", lenient=False)
            else:
                privacy_spent=None
            print('Epoch: ', epoch, "total: ", len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # print("Privacy_Cost: ",privacy_spent) # the same as train
        
        # save final results of testing 
        test_final_result={
            "mode":"test",
            "epoch:":epoch,
            "test_loss":test_loss/(batch_idx+1),
            "test_acc":correct/total,
            "eps_rdp":privacy_spent["eps_rdp"] if 'nonDP' not in args.clipping_mode else 'nonDP',
            "alpha_rdp":privacy_spent["alpha_rdp"] if 'nonDP' not in args.clipping_mode else 'nonDP',
            "eps_low":privacy_spent["eps_low"] if 'nonDP' not in args.clipping_mode else 'nonDP',
            "eps_estimate":privacy_spent["eps_estimate"] if 'nonDP' not in args.clipping_mode else 'nonDP',
            "eps_upper":privacy_spent["eps_upper"] if 'nonDP' not in args.clipping_mode else 'nonDP',
        }
        save_json(test_final_result,FILE_PATH+"final_results.json")


    def get_eval_results()->dict:# 由于没找到验证集，实际上返回的是测试集的结果
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        net.train() # 由于在train()内部调用，需要重置为train模式
        return {
            "eval_loss":test_loss/(batch_idx+1),
            "eval_acc":correct/total,
        }


    for epoch in range(args.epochs):
        train(epoch,privacy_engine)
        test(epoch,privacy_engine)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=3, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--mini_bs', type=int, default=100)
    parser.add_argument('--epsilon', default=2, type=float, help='target epsilon')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max_grad_norm denoted by C')
    parser.add_argument('--delta', default=1e-5, type=float, help='target delta')
    parser.add_argument('--clipping_mode', default='BK-MixOpt', type=str)
    parser.add_argument('--clipping_style', default='all-layer', nargs='+',type=str)
    parser.add_argument('--model', default='vit_small_patch16_224', type=str)
    parser.add_argument('--cifar_data', type=str, default='CIFAR10')
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--note', type=str, default=None,help="notes of the experiment")
    parser.add_argument('--exp_name', type=str, default=None,help="specific name of the experiment")
    parser.add_argument('--save_log', type=int, default=1) # boolen doesn't work..

    args = parser.parse_args()
    
    from fastDP import PrivacyEngine

    import os 
    import json
    from datetime import datetime

    import torch
    import torchvision
    torch.manual_seed(2)
    import torch.nn as nn
    import torch.optim as optim
    import timm
    from opacus.validators import ModuleValidator
    from opacus.accountants.utils import get_noise_multiplier
    from tqdm import tqdm
    import warnings; warnings.filterwarnings("ignore")

    main(args)
