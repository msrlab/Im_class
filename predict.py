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
   
### load the checkpoint
model, optimizer, log = load_checkpoint(model, optimizer, f"{save_dir}/{model_str}_last_epoch.pth", device)

##### needs to be adapted, move to seperate file with functions for label handling?

idx_to_class = {val: key for key, val in fl_model.class_to_idx.items()} #this line adapted from the Udacity Knowledge base
def idx_to_name (idx):
    fclass=idx_to_class[idx]
    name = cat_to_name[fclass]
    return name

####show prediction results with own picture
#filename='Flower_example4.jpg'
#fig, axs = pypl.subplots(nrows=2, ncols=1)
#im = process_image(filename)
t_prob, t_class = predict(filename,fl_model,5)
print(t_prob)
print(t_class)
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