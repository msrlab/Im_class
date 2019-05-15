### helper functions for image processing, visualization etc.


#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = pypl.subplots()
        pypl.title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def process_image(image):
    
    def normalize (x, mean, std):
        x_scale = (x - np.mean(x)) / 255
        x_norm = (x_scale-np.mean(x_scale))/std + mean
        return x_norm

    im = Image.open(image)
    norm_size = tuple([int(x/(min(im.size)/256)) for x in im.size])
    im = im.resize(norm_size, resample=Image.LANCZOS)
    w, h = im.size
    l = (w - 224)/2
    t = (h - 224)/2
    r = (w + 224)/2
    b = (h + 224)/2
    im = im.crop((l, t, r, b))
    #display(im)
    imnp = np.asarray(im)
    channel1 = normalize(imnp[:,:,0], 0.485, 0.229)
    channel2 = normalize(imnp[:,:,1], 0.456, 0.224)
    channel3 = normalize(imnp[:,:,2], 0.406, 0.225)
    im_tensor = torch.Tensor(3,224,224)
    im_tensor[0] = torch.from_numpy(channel1)
    im_tensor[1] = torch.from_numpy(channel2)
    im_tensor[2] = torch.from_numpy(channel3)
    return im_tensor