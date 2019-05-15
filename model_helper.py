# functions to setup the model
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import time
#import json


def construct_nn_Seq (nr_in_features, hl_nodes, nr_out_features, out_function, p_dropout):
    #add first hidden layer
    if len(hl_nodes) == 0:
        d = OrderedDict([('cl_1', nn.Linear(nr_in_features,nr_out_features))]) 
    else:
        #add first hidden layer
        d = OrderedDict([('cl_1', nn.Linear(nr_in_features,hl_nodes[0])),
                        ('relu_1', nn.ReLU()),
                        ('dropout_1', nn.Dropout(p_dropout[0])) ])
        nr_hl = len(hl_nodes)
        #add next hidden layer(s)
        if nr_hl > 1:
            for i in range(nr_hl-1):
                d['cl_'+str(i+2)] = nn.Linear(hl_nodes[i],hl_nodes[i+1])
                d['relu_'+str(i+2)] = nn.ReLU()
                d['dropout_'+str(i+2)] = nn.Dropout(p_dropout[i+1])
        else:
            i = 0
        #add last layer    
        d['cl_'+str(nr_hl+1)] = nn.Linear(hl_nodes[nr_hl-1],nr_out_features)
    d['out'] = out_function
    classifier = nn.Sequential(d)
    return classifier

def setup_model(mp):
    #load pretrained model & get required number of input features
    if mp.model_family == 'vgg16':  
        model = models.vgg16(pretrained=True)
        nr_in_features = model.classifier[0].in_features
    if mp.model_family == 'vgg19':  
        model = models.vgg19(pretrained=True)
        nr_in_features = model.classifier[0].in_features
    if mp.model_family == 'densenet121':  
        model = models.densenet121(pretrained=True)
        nr_in_features = model.classifier.in_features
        
    # attach labels to model
    model.class_to_idx = mp.class_to_idx
    
    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False  
        
    #setup layers in classifier and add to model
    classifier = construct_nn_Seq (nr_in_features, mp.hl_nodes, mp.nr_out_features, mp.out_function, mp.p_dropout)    
    model.classifier = classifier

    return model


####### functions and classes for logging, saving and loading
def save_checkpoint(model, optimizer, filename, train_logger):
    checkpoint = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'train_log': train_logger,
                 'parameters': model.parameters}
    torch.save(checkpoint, filename)
    
def load_checkpoint(model, optimizer, filename, device):
    if device.type == 'cpu':
        checkpoint = torch.load(filename, map_location='cpu')
    else:
        checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.parameters =  checkpoint['parameters']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_log = checkpoint['train_log']
    return model, optimizer, train_log  

#structure for collecting training stats and saving
class train_logger:
    def __init__(self):
        self.val_acc = []
        self.val_loss = []
        self.train_loss = []
        self.epochs = 0
   
#structure for saving model parameters for reconstruction
class model_setup_parms():
    def __init__(self):
        self.model_family = ''
        self.hl_nodes = []
        self.p_dropout = []
        self.nr_out_features = None
        self.out_function = None
        self.class_to_idx = None
                        

###### functions to calculate and print metrics

def calc_val_metrics (device, fl_model, dataloader_valid, criterion):
    start_time_eval = time.time()
    fl_model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for jj, (inputs, labels) in enumerate(dataloader_valid):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = fl_model.forward(inputs)
            val_loss += criterion(outputs,labels) 
            #accuracy
            prob = torch.exp(outputs)
            top_p, top_class = prob.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            val_accuracy += torch.mean(equals.type(torch.FloatTensor))
            #test_acc = torch.mean(equals.type(torch.FloatTensor))
            #print(f"test_acc {test_acc:.3f}")
        val_accuracy = val_accuracy / (jj+1)
        val_loss = val_loss / (jj+1)
        val_time = time.time() - start_time_eval
    fl_model.train()
    return val_time, val_loss, val_accuracy


##todo make more generic, just give a dictionary to print out arbitrary information
def print_loss_metrics(epoch, epochs, batch, val_accuracy, val_loss, train_loss, time):  
    print(f"Epoch {epoch}/{epochs} | "
          #f"Batch {ii} batch_time:{batch_time:.3f}s | "
          #f"val time {val_time:.3f} | "
          f"val accuracy {val_accuracy:.3f} | "
          f"val loss {val_loss:.3f} | "
          f"train loss {train_loss:.3f} | "
          f"time {time:.3f} | ")

    
def train(dataloader, model_str, device, model, optimizer, criterion, epochs, eval_every_x_batch, eval_every_x_epoch, save_every_x_evalepoch, resume, save_dir):    
    # initialize
    print("Starting training...")
    if device.type == 'cpu':
        print("Warning: training in CPU mode may take a veeeeeeeeeeeeeeeeeeery long time.")
    if device.type == 'cuda':
        print("GPU mode enabled")     
    model.to(device)
    log = train_logger()
    resume_epoch = 0

    #load checkpoint to continue training
    if os.path.isfile(f"{save_dir}/{model_str}_last_epoch.pth"):
        model, optimizer, log = load_checkpoint(model, optimizer, f"{save_dir}/{model_str}_last_epoch.pth", device)
        resume_epoch = log.epoch
        print(f"Checkpoint detected: Resuming from {save_dir}/{model_str}_last_epoch.pth")
        print_loss_metrics(log.epoch, epochs, 0, log.val_acc[log.epoch - 1], log.val_loss[log.epoch - 1], log.train_loss[log.epoch - 1], 0)
        if resume_epoch == epochs:
            print(f"training on {epochs} epochs is already finished, increase nr of epochs if you want to continue training")
        elif resume_epoch > epochs:
            print(f"more than {epochs} already trained, increase nr of epochs if you want to continue training")
        else:
            print("Continuing...")
    else:
        print("no checkpoint detected, starting from scratch....")
    for epoch in range(resume_epoch, epochs):
        start_time_e = time.time()
        loss_running = 0
        for ii, (inputs, labels) in enumerate(dataloader['train']):    
            optimizer.zero_grad()
            start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs= model.forward(inputs)
            loss = criterion(outputs,labels)
            loss_running += loss
            loss.backward()
            optimizer.step()
            batch_time = time.time()-start_time
            #print(f"Batch {ii} batch_time:{batch_time:.3f}s "
            #      f"train loss {loss:.3f}")
            if (ii+1) % eval_every_x_batch == 0:
                val_time, val_loss, val_accuracy = calc_val_metrics(device, model, dataloader['valid'], criterion)
                log.val_acc.append(val_accuracy)
                log.val_loss.append(val_loss)
                log.train_loss.append(loss_running/(ii+1))
                log.epoch = epoch + 1
                print_loss_metrics(epoch, epochs, ii+1, val_accuracy, val_loss, loss_running/(ii+1))
        if (epoch+1) % eval_every_x_epoch == 0:  #check validation set and print metrics
            epoch_time = time.time() - start_time_e 
            val_time, val_loss, val_accuracy = calc_val_metrics(device, model, dataloader['valid'], criterion)
            log.val_acc.append(val_accuracy)
            log.val_loss.append(val_loss)
            log.train_loss.append(loss_running/(ii+1))
            log.epoch = epoch + 1
            print_loss_metrics(epoch+1, epochs, ii+1, val_accuracy, val_loss, loss_running/(ii+1), epoch_time)
            if (epoch+1) % save_every_x_evalepoch == 0: #save checkpoint for later resuming
                save_checkpoint(model, optimizer, f"{save_dir}/{model_str}_last_epoch.pth", log)           
                #later implement additional saving of model with lowest loss  f"{model_str}_epoch{epoch+1:03d}.pth"
    return log
    # Tracking the loss and accuracy on the validation set to determine the best hyperparameters
    
    
