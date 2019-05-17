import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
from model_helper import load_checkpoint_reconstruct, predict_im, calc_val_metrics
from image_helper import process_image
import argparse
from data_helper import load_labels, make_dataloader 
import numpy as np
  
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store',
                    help='directory containing the data. must contain test, valid and train subfolders. these must contain subfolders for each category.')
parser.add_argument('checkpoint', action='store',
                    help='checkpoint of trained model')
parser.add_argument('--top_k', action='store', type=int,
                    dest='top_k', default=5,
                    help='output of top k classes')
parser.add_argument('--nr_probes', action='store', type=int,
                    dest='nr_probes', default=5,
                    help='do classification for this many images for a random sample from each dataset (train, test, valid)')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='set_gpu',
                    help='switch to set gpu mode explicitely. default is autodetect')
parser.add_argument('--cpu', action='store_true',
                    default=False,
                    dest='set_cpu',
                    help='switch to set cpu mode explicitely. default is autodetect')
parser.add_argument('--printmodel', action='store_true',
                    default=False,
                    dest='printmodel',
                    help='for debugging: print model architecture to console')
parser.add_argument('--performance', nargs = '+', action='store',
                    default=[],
                    dest='performance',
                    help='calculate overall performance (accuracy) for data sets. can have multiple options: train test valid. Caution: may take a long time for train set, and even longer in cpu mode')

args = parser.parse_args()
data_dir = args.data_dir
nr_probes = args.nr_probes
checkpoint = args.checkpoint
top_k = args.top_k
printmodel = args.printmodel
set_cpu = args.set_cpu
set_gpu = args.set_gpu
acc_dsets = args.performance

if set_gpu:
    device = torch.device('cuda:0')
    print("Device manually set to cuda")
elif set_cpu:
    device = torch.device('cpu')
    print("Device manually set to cpu")
else:  #autodetect
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device autodetected as {device.type}")

fl_model, log = load_checkpoint_reconstruct(checkpoint, device)
fl_model.to(device)
if printmodel:
    print(fl_model)

dataloader, class_to_idx = make_dataloader(data_dir) 
dsets = ['train', 'valid', 'test']
criterion = nn.NLLLoss()
fl_model.eval()
dataiter = {x: iter(dataloader[x]) for x in dsets}
np.set_printoptions(precision = 3)
for dset in dsets:   
    images, labels = next(dataiter[dset])
    if dset in acc_dsets:
        print(f"calculating performance on {dset} set...")
        val_time, test_loss, test_accuracy = calc_val_metrics (device, fl_model, dataloader[dset], criterion)
        print(f"accuracy:{test_accuracy:.3f}")
    print(f"checking true label against prediction for {dset} set")
    for i in range(nr_probes):    
        t_prob, t_class = predict_im(images[i],fl_model,device,top_k)
        t_class
        print(f"true label:{labels[i]:03d} | prediction:{t_class[0]:03d} | {t_class} | {t_prob}") 
