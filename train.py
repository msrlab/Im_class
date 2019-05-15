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

#select the model including parameters
def select_model(select_model):
    nr_out_features = len(cat_to_name)
    if select_model == 'densenet121_a':
        fl_model = setup_model('densenet121', [1024, 512, 256, 256], [0.2, 0.2, 0.2, 0.2], nr_out_features, nn.LogSoftmax(dim=1), class_to_idx)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(fl_model.classifier.parameters(), lr=0.0001)
    if select_model == 'densenet121_test':
        fl_model = setup_model('densenet121', [512, 256], [0.2, 0.2], nr_out_features, nn.LogSoftmax(dim=1), class_to_idx)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(fl_model.classifier.parameters(), lr=0.0001)

    if select_model == 'densenet121_b':
        fl_model = setup_model('densenet121', [512, 256], [0.5, 0.5], nr_out_features, nn.LogSoftmax(dim=1), class_to_idx)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(fl_model.classifier.parameters(), lr=0.0001)   

    if select_model == 'vgg19_a':
        fl_model = setup_model('vgg19', [1600, 1600], [0.5, 0.5], nr_out_features, nn.LogSoftmax(dim=1), class_to_idx)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(fl_model.classifier.parameters(), lr=0.0001) 

    if select_model == 'vgg16_a':
        fl_model = setup_model('vgg16', [1600, 1600], [0.5, 0.5], nr_out_features, nn.LogSoftmax(dim=1), class_to_idx)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(fl_model.classifier.parameters(), lr=0.001) 
        #optimizer = optim.SGD(fl_model.parameters(), lr = 0.005, momentum = 0.5)
        #optimizer = optim.ASGD( fl_model.classifier.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

    return fl_model, criterion, optimizer
#print(fl_model)



###### load the data
data_dir = 'flowers'
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


#########  do the training

model_str = 'densenet121_test'
fl_model, criterion, optimizer = select_model(model_str)
####train the selected model
# parameters and settings for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'  #or set manually to 'cpu' or 'cuda'
fl_model.to(device)
resume = True #True to resume from saved checkpoint !!!caution: if set to False, existing file will be overwritten!
epochs = 30
eval_every_x_batch = 1000    #can be used for testing in slow CPU mode, set to high values if not needed
eval_every_x_epoch = 1
save_every_x_evalepoch = 1 * eval_every_x_epoch

with active_session():
    log = train(dataloader, model_str, device, fl_model, optimizer, criterion, epochs, eval_every_x_batch, eval_every_x_epoch, save_every_x_evalepoch, resume)
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
pypl.savefig('Loss')