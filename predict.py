import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pypl
import torch
import os
from model_helper import load_checkpoint_reconstruct, predict_im
from image_helper import process_image
import argparse
import numpy as np
from data_helper import load_labels
from msr_helper import imshow

def idx_to_name (model, idx):
    fclass=model.idx_to_class[idx]
    name = cat_to_name[fclass]
    return name

def plot_prediction (image, label, t_names, t_prob, save_fname):
    fig, axs = pypl.subplots(nrows=2, ncols=1)
    imshow(image, ax = axs[0], title=label)
    axs[0].title.set_text(label)
    axs[1].barh([str(x) for x in t_names], t_prob)
    axs[1].barh([str(x) for x in t_names], t_prob)
    pypl.savefig(save_fname)

parser = argparse.ArgumentParser()
parser.add_argument('path_to_image', action='store',
                    help='image to be classified')
parser.add_argument('checkpoint', action='store',
                    help='checkpoint of trained model')
parser.add_argument('--top_k', action='store', type=int,
                    dest='top_k', default=5,
                    help='output of top k classes')
parser.add_argument('--category_names', action='store', 
                    dest='category_names', default='cat_to_name.json',
                    help='json dictionary to translate classes/categories to names')

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
parser.add_argument('--png_print', action='store_true',
                    default=False,
                    dest='png_print',
                    help='print a nice prediction graphic ot predict.png')

args = parser.parse_args()
cat_names_file = args.category_names
image = args.path_to_image
checkpoint = args.checkpoint
top_k = args.top_k
printmodel = args.printmodel
set_cpu = args.set_cpu
set_gpu = args.set_gpu
png_print = args.png_print

# set device cpu/gpu
if set_gpu:
    device = torch.device('cuda:0')
    print("Device manually set to cuda")
elif set_cpu:
    device = torch.device('cpu')
    print("Device manually set to cpu")
else:  #autodetect
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device autodetected as {device.type}")

### load the checkpoint and reconstruct model
fl_model, log = load_checkpoint_reconstruct(checkpoint, device)
fl_model.to(device)
if printmodel:
    print(fl_model)

#load image and predict
in_im = process_image(image)
t_prob, t_class = predict_im(in_im,fl_model,device,top_k)
np.set_printoptions(precision = 3)
print(f"class probabilities: {t_prob}")
print(f"class index: {t_class}")
if os.path.isfile(cat_names_file):
    #get category-to-label mapping
    cat_to_name = load_labels(cat_names_file)
    t_names = [idx_to_name(fl_model,x) for x in t_class]
    print(t_names)
    if png_print:
        plot_prediction(in_im, image, t_names, t_prob, 'predict')
else: 
    print(f"'{cat_names_file}' does not exist.")


    
