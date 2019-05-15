print("importing libraries...")
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pypl
import msr_helper as helper
from workspace_utils import active_session
from PIL import Image
import numpy as np
from IPython.display import display
from model_helper import construct_nn_Seq, setup_model, save_checkpoint, load_checkpoint, train_logger, calc_val_metrics, print_loss_metrics, train
from proc_helper import process_image
import argparse

# parameters and settings for training

#defaults for command line arguments
def_epochs = 30
def_save_dir = '.'
def_hidden_units = [512]
def_dropout = [0.2]
def_learning_rate = 0.0001

resume = True #True to resume from saved checkpoint !!!caution: if set to False, existing file will be overwritten!

eval_every_x_batch = 1000    #can be used for testing in slow CPU mode, set to high values if not needed
eval_every_x_epoch = 1
save_every_x_evalepoch = 1 * eval_every_x_epoch
model_str = 'densenet121_test'

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store',
                    help='directory containing the data. must contain test, valid and train subfolders. these must contain subfolders for each category.')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir', default=def_save_dir,
                    help='directory where checkpoints and other stuff is saved. default is current directory. also used to resume training.')
parser.add_argument('--arch', action='store',
                    dest='arch',
                    help='choose architecture of pretrained network. available options: vgg19, densenet212. take a look at setup_model() to implement more')
parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate', default=def_learning_rate,
                    help='give learning rate for training')
parser.add_argument('--epochs', action='store', type=int,
                    dest='epochs', default=def_epochs,
                    help='how many epochs to be trained')
parser.add_argument('--hidden_units', nargs ='+', action='store', type=int,
                    dest='hidden_units', default=def_hidden_units,
                    help='number of hidden units per layer. can be multiple arguments, each additional number adds a fully connected layer to the classifier')
parser.add_argument('--dropout', nargs ='+', action='store', type=float,
                    dest='dropout', default=def_dropout,
                    help='dropout used during training. can be given as single number or per layer')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='set_gpu',
                    help='switch to set gpu mode explicitely. default is autodetect')
parser.add_argument('--cpu', action='store_true',
                    default=False,
                    dest='set_cpu',
                    help='switch to set cpu mode explicitely. default is autodetect')
parser.add_argument('--noresume', action='store_true',
                    default=False,
                    dest='noresume',
                    help='default behavior is to resume training from saved checkpoint in <save_dir>. this switch will override resuming and overwrite any saved checkpoint')
parser.add_argument('--printmodel', action='store_true',
                    default=False,
                    dest='printmodel',
                    help='for debugging: print model architecture to console')

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
epochs = args.epochs
model_str = args.arch
hidden_units = args.hidden_units
dropout = args.dropout
learning_rate = args.learning_rate
printmodel = args.printmodel
set_cpu = args.set_cpu
set_gpu = args.set_gpu

## if only one dropout value but multiple layers in hidden_units, construct p_dropout list with same value
if (len(dropout) == 1):
    p_dropout = [dropout[0] for i in range(len(hidden_units))]
else:
    p_dropout = dropout
## makedirectory if not exist
try:
    os.mkdir(save_dir)
except FileExistsError:
    pass
## set gpu/cpu mode
if set_gpu:
    device = torch.device('cuda:0')
    print("Device manually set to cuda")
elif set_cpu:
    device = torch.device('cpu')
    print("Device manually set to cpu")
else:  #autodetect
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device autodetected as {device.type}")

###### load the data
#data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

###### setting parameters
par_rot = 45 #max_rotation to apply
par_imsize = 255  #resize to
par_crop = 224  #crop to
par_norm = {'mean': [0.485, 0.456, 0.406],    #normalize RGB values to
            'std': [0.229, 0.224, 0.225]} 
par_bsize = 128

# define transforms for training, validation, and testing sets
data_transforms = {
                  'train': transforms.Compose([                   
                     transforms.RandomRotation(par_rot),
                     transforms.Resize(par_imsize),
                     transforms.RandomVerticalFlip(),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomResizedCrop(par_crop),
                     transforms.ToTensor(),
                     transforms.Normalize(par_norm['mean'],par_norm['std'])]),
                  'test': transforms.Compose([
                     transforms.Resize(par_imsize),
                     transforms.CenterCrop(par_crop),
                     transforms.ToTensor(),
                     transforms.Normalize(par_norm['mean'],par_norm['std'])])
                    }
data_transforms['valid'] = data_transforms['train']

# Load the datasets with ImageFolder
print(f"loading the dataset from folder '{data_dir}' ...")
dsets = ['train','valid','test']
image_dataset = {x: datasets.ImageFolder(data_dir + '/' + x, data_transforms[x])
                     for x in dsets}

#  definition of dataloaders
dataloader = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size = par_bsize, shuffle = True)
                  for x in dsets}
dataiter = {x: iter(dataloader[x]) for x in dsets}
test = next(dataiter['train'])
class_to_idx = image_dataset['train'].class_to_idx
#for debugging and checking:  show sample images  (adapted from Udacity pytorch lesson part 7)
#for x in dsets:
#    images, labels = next(dataiter[x])
#    fig, axes = pypl.subplots(figsize=(10,4), ncols=4)
#    for ii in range(4):
#        ax = axes[ii]
#        #######change to a generic imshow method
#        helper.imshow(images[ii], ax=ax, normalize=True)
        
######## load label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
len(cat_to_name)

### setup model
nr_out_features = len(cat_to_name)
fl_model = setup_model(model_str, hidden_units, p_dropout, nr_out_features, nn.LogSoftmax(dim=1), class_to_idx)
criterion = nn.NLLLoss()
optimizer = optim.Adam(fl_model.classifier.parameters(), lr=learning_rate)
if printmodel:
    print(fl_model)


        #optimizer = optim.SGD(fl_model.parameters(), lr = 0.005, momentum = 0.5)
        #optimizer = optim.ASGD( fl_model.classifier.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

#print(fl_model)




####train the selected model
fl_model.to(device)

with active_session():
    log = train(dataloader, model_str, device, fl_model, optimizer, criterion, epochs, eval_every_x_batch, eval_every_x_epoch, save_every_x_evalepoch, resume, save_dir)
print(f"finished training on {epochs} epochs.")
print("calculating performance on test set...")
fl_model.eval()
val_time, test_loss, test_accuracy = calc_val_metrics (device, fl_model, dataloader['test'], criterion)
print(f"Accuracy:{test_accuracy:.3f}")
    
  ######### ToDo make dependent on command line or certain state  
#just load the last checkpoint of training
#modelstr = 'vgg19_a'
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#fl_model, criterion, optimizer = select_model(modelstr)
#fl_model, optimizer, train_log = load_checkpoint(fl_model, optimizer, modelstr+'_last_epoch.pth')
#fl_model = fl_model.to(device)


######## ToDO transform into a function 
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
pypl.plot(log.val_loss, label='Validation Loss')
pypl.plot(log.train_loss, label='Train Loss')
pypl.legend()
pypl.savefig(save_dir+'/Loss')