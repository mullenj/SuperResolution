import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from vdsr import Net
from sklearn.model_selection import train_test_split
import numpy as np

im_dir = '/media/me-user/Backup Plus/training_data_2'

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

class npDataset(Dataset):
    """
    npDataset will take in the list of numpy files and create a torch dataset
    out of it.
    """
    def __init__(self, data_list, batch_size):
        self.array = data_list
        self.batch_size = batch_size
        
    def __len__(self): return int((len(self.array)/self.batch_size))
    
    def __getitem__(self, i):
        """
        getitem will first select the batch of files before loading the files
        and splitting them into the goes and viirs components, the input and target
        of the network
        """
        files = self.array[i*self.batch_size:i*self.batch_size + self.batch_size]
        x = []
        y = []
        for file in files:
            file_path = os.path.join(im_dir, file)
            sample = np.load(file_path)
            x.append(sample[:,:,1])
            y.append(sample[:,:,0])
        x, y = np.array(x)/255., np.array(y)/255.
        x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
        return torch.Tensor(x), torch.Tensor(y)

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    file_list = os.listdir(im_dir)
    print(f'{len(file_list)} data samples found')
    train_files, test_files = train_test_split(file_list, test_size = 0.2, random_state = 42)
    train_loader = DataLoader(npDataset(train_files, opt.batch_size))
    test_loader = DataLoader(npDataset(test_files, opt.batch_size))

    print("===> Building model")
    model = Net()
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    model = model.cuda()
    criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(train_loader, test_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, testing_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, (x,y) in enumerate(training_data_loader):
        #print('**********')
        input, target = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
        #print(input.size(), target.size())

        input = input.cuda()
        target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()

        #if iteration%1 == 0:
        print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))
        #print('##########')
    with torch.no_grad():
        for x, y in testing_data_loader:
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            x, y = x.cuda(), y.cuda()
            test_loss = criterion(model(x), y)
            #print('%%%%%%%%%%')
    test_loss /= len(testing_data_loader.dataset)        
    print(f'Test Reconstruction Loss: ({test_loss.item()})]')

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
