print("importing libraries...")
import matplotlib
matplotlib.use('Agg')
import os
import torch
from torchvision import models
from torch import nn
from torch import optim
from workspace_utils import active_session
from model_helper import construct_nn_Seq, setup_model, calc_val_metrics, train, model_setup_parms
from data_helper import make_dataloader, plot_loss 
import argparse

#defaults for command line arguments
def_epochs = 30
def_save_dir = '.'
def_hidden_units = [512, 256]
def_dropout = [0.2]
def_learning_rate = 0.0001
def_arch = 'densenet121'
# settings
eval_every_x_epoch = 1   #useful to set to higher numpers if many epochs are needed for training and calculation of validaiton accuacy takes a long time
save_every_x_evalepoch = 1 * eval_every_x_epoch   

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store',
                    help='directory containing the data. must contain test, valid and train subfolders. these must contain subfolders for each category.')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir', default=def_save_dir,
                    help='directory where checkpoints and other stuff is saved. default is current directory. also used to resume training.')
parser.add_argument('--arch', action='store',
                    dest='arch', default=def_arch,
                    help='choose architecture of pretrained network. available options: vgg19, densenet212. take a look at setup_model() to implement more')
parser.add_argument('--learning_rate', action='store', type = float,
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
                    help='default behavior is to resume training from saved checkpoint in <save_dir>. this switch will override resuming and overwrite any saved checkpoint. Caution, resuming assumes that the same model is given!')
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
resume = not(args.noresume)

## if only one dropout value but multiple layers in hidden_units, construct p_dropout list with same value
if (len(dropout) == 1):
    p_dropout = [dropout[0] for i in range(len(hidden_units))]
else:
    p_dropout = dropout
    
## make save_dir directory if not exist
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

### setup model
nr_out_features = len(class_to_idx)
mp = model_setup_parms()
mp.model_family, mp.hl_nodes, mp.p_dropout, mp.nr_out_features, mp.out_function, mp.class_to_idx = model_str, hidden_units, p_dropout, nr_out_features, nn.LogSoftmax(dim=1), class_to_idx
fl_model = setup_model(mp)
fl_model.mp = mp  #needs to be attached to model for saving into checkpoint and do a complete model reconstruction when loading a checkpoint
criterion = nn.NLLLoss()
optimizer = optim.Adam(fl_model.classifier.parameters(), lr=learning_rate)
if printmodel:
    print(fl_model)

####train the selected model
fl_model.to(device)
with active_session():
    log = train(dataloader, model_str, device, fl_model, optimizer, criterion, epochs, eval_every_x_epoch, save_every_x_evalepoch, resume, save_dir)
print(f"finished training on {epochs} epochs.")
## some output information
plot_loss(log.val_loss, log.train_loss, save_dir, model_str)
print(f"see {save_dir}/Loss_{model_str}.png for visualization of training and validation loss by epoch")       
print("calculating performance on test set...")
fl_model.eval()
val_time, test_loss, test_accuracy = calc_val_metrics (device, fl_model, dataloader['test'], criterion)
print(f"test accuracy:{test_accuracy:.3f}")
best_epoch = log.val_acc.index(max(log.val_acc))+1
print(f"best validation accuracy of {max(log.val_acc):0.3f} at epoch {best_epoch} (checkpoint:'{save_dir}/{model_str}_best.pth'), checkout test accuracy for this model (checkout.py)") 