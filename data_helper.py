import torch
import json
from torchvision import transforms, datasets

def make_dataloader(data_dir):
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
    #dataiter = {x: iter(dataloader[x]) for x in dsets}
    class_to_idx = image_dataset['train'].class_to_idx
    return dataloader, class_to_idx


######## load label mapping
def load_labels(cat_names_file):
    with open(cat_names_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
