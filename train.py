print("importing libraries...")
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pypl
import msr_helper as helper
from workspace_utils import active_session
from PIL import Image
import numpy as np
from IPython.display import display
from model_helper import construct_nn_Seq, setup_model, save_checkpoint, load_checkpoint, train_logger, calc_val_metrics, print_loss_metrics, train, model_setup_parms
from proc_helper import process_image
from data_helper import load_labels, make_dataloader 
import argparse

# parameters and settings for training

#defaults for command line arguments
def_epochs = 30
def_save_dir = '.'
def_hidden_units = [512]
def_dropout = [0.2]
def_learning_rate = 0.0001
def_arch = 'densenet121'

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
                    dest='arch', default=def_arch,
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
    
#construct dataloader
dataloader, class_to_idx = make_dataloader(data_dir)    
#get category-to-label mapping
cat_to_name = load_labels()

### setup model
nr_out_features = len(cat_to_name)
mp = model_setup_parms()
mp.model_family, mp.hl_nodes, mp.p_dropout, mp.nr_out_features, mp.out_function, mp.class_to_idx = model_str, hidden_units, p_dropout, nr_out_features, nn.LogSoftmax(dim=1), class_to_idx
fl_model = setup_model(mp)
fl_model.parameters = mp
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