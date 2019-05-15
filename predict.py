import torch
from model_helper import load_checkpoint_reconstruct
from image_helper import process_image
import argparse
from data_helper import load_labels
#predict from input image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    fl_model.eval()
    in_im = process_image(image_path)
    inputs = torch.Tensor(1,3,224,224)
    inputs = inputs.to(device)
    inputs[0] = in_im
    #print(inputs)
    output = fl_model.forward(inputs)
    #print(output)
    prob = torch.exp(output)
    top_prob, top_class = torch.topk(prob, topk) 
    return top_prob.cpu().detach().numpy()[0], top_class.cpu().detach().numpy()[0]
    # TODO: Implement the code to predict the class from an image file

    
def predict2(in_im, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    fl_model.eval()
    #in_im = process_image(image_path)
    inputs = torch.Tensor(1,3,224,224)
    inputs = inputs.to(device)
    inputs[0] = in_im
    #print(inputs)
    output = fl_model.forward(inputs)
    #print(output)
    prob = torch.exp(output)
    top_prob, top_class = torch.topk(prob, topk) 
    return top_prob.cpu().detach().numpy()[0], top_class.cpu().detach().numpy()[0]
    # TODO: Implement the code to predict the class from an image file   
   
def idx_to_name (model, idx):
    fclass=model.idx_to_class[idx]
    name = cat_to_name[fclass]
    return name


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

args = parser.parse_args()
cat_names_file = args.category_names
image = args.path_to_image
checkpoint = args.checkpoint
top_k = args.top_k
printmodel = args.printmodel
set_cpu = args.set_cpu
set_gpu = args.set_gpu

#get category-to-label mapping
cat_to_name = load_labels(cat_names_file)

### load the checkpoint

if set_gpu:
    device = torch.device('cuda:0')
    print("Device manually set to cuda")
elif set_cpu:
    device = torch.device('cpu')
    print("Device manually set to cpu")
else:  #autodetect
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device autodetected as {device.type}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#checkpoint = 'densenet121_last_epoch.pth'
#filename = 'probe.jpg'
fl_model, log = load_checkpoint_reconstruct(checkpoint, device)
fl_model.to(device)
if printmodel:
    print(fl_model)
   

####show prediction results with own picture
#filename='Flower_example4.jpg'
#fig, axs = pypl.subplots(nrows=2, ncols=1)
#im = process_image(filename)
t_prob, t_class = predict(image,fl_model,top_k)
print(t_prob)
#print(t_class)
t_names = [idx_to_name(fl_model,x) for x in t_class]
print(t_names)
#imshow(im, ax = axs[0])
#axs[1].barh([idx_to_name(x) for x in t_class], t_prob)
#axs[0].title.set_text(filename)


#### TODO move to different file and recreate as a function
###sanity check with labeled images
#images, labels = next(dataiter['test'])
#bla = iter(range(len(images)-1))
#
#i = next(bla)
#fig, axs = pypl.subplots(nrows=2, ncols=1)
#t_prob, t_class = predict2 (images[i],fl_model,5)
#imshow(images[i], ax = axs[0], title=labels[i].numpy())
#axs[0].title.set_text(idx_to_name(np.ndarray.tolist(labels[i].numpy())))
#axs[1].barh([idx_to_name(x) for x in t_class], t_prob)