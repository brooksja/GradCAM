import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import os
from torchvision.models import resnet18,ResNet18_Weights
from fastai.learner import load_learner
from PIL import Image

def GCAM(img_path:Path,
         MILmodel_path:Path,
         extractor:nn.Module=None,
         layer:int=5,
         transforms = None,
         outpath:Path = Path(os.getcwd())):
    """
    Main function; preps models and input image, runs image through models, gets activations and plots results
    inputs:
        img_path - path to image to be analysed
        MILmodel_path - path to saved MIL model checkpoint
        extractor - extractor model
        layer - layer of interest in the extractor
        transform - image preprocessing transforms
        outpath - path for saving output
    returns:
        activations - layer activations at the layer of interest
    outputs:
        GCAM_layer_<layer>.png - side-by-side plot of image and acivations
    """

    if not str(MILmodel_path).endswith('.pkl'):
        print('Specify path to MIL model .pkl object')
        raise TypeError()
    
    if not extractor:
        extractor,transforms = default_extractor()
    if not transforms:
        _,transforms = default_extractor()

    # Create save path for output
    img_name = os.path.basename(os.path.splitext(img_path)[0])
    save_path = os.path.join(outpath,img_name,f'GCAM_layer_{layer}.png')
    os.makedirs(save_path,exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor.to(device)

    # Split extractor to access layer of interest
    extractor_layers = [layer for layer in extractor.children()]
    extractor_start = nn.Sequential(*extractor_layers[:layer])
    extractor_end = nn.Sequential(*extractor_layers[layer:-1])
    extractor_last = extractor_layers[-1]

    # Load MIL model from provided .pkl file
    learn = load_learner(MILmodel_path)
    MILmodel = learn.model.eval().to(device)

    # Load image and apply transforms
    orig = Image.open(img_path)
    img = transforms(orig).unsqueeze(0).to(device)

    # Put image through the extractor to get feats
    x = extractor_start(img)
    x.retain_grad() # keep the gradients from the layer of interest
    z = extractor_end(x)
    z.squeeze().unsqueeze(0) # force correct shape
    feats = extractor_last(z)

    # Put feats through MIL model
    z = MILmodel.encoder(feats)
    attention = MILmodel._masked_attention_scores(z,torch.tensor(z.shape[1],device=device))
    z = (attention*z).sum(-2)
    y = MILmodel.head(z)

    # Do back propagation
    class_idx = torch.argmax(y[0]) # finds index of predicted class
    y[0,class_idx].backward()
    activations = nn.functional.relu(x*x.grad).squeeze().sum(0).detach().cpu()

    # Plot and save
    plot_results(orig,activations,save_path)

    # Return the activations in case user wants them for anything
    return activations

def default_extractor():
    """
    Function to load a default model if no extractor is provided. Also loads the ImageNet transforms
    returns:
    resnet18, ImageNet transforms
    """
    # Use default weights from torchvision
    weights = ResNet18_Weights.DEFAULT
    # Load the default (ImageNet) transforms
    transforms = weights.transforms
    # Load ResNet18 model
    model = resnet18(weights=weights,progress=False)
    # Replace final (fully-connected) layer with identity to get features
    model.fc = nn.Identity()
    return model,transforms

def plot_results(img,act,save_path):
    """
    Function to plot the original image and GradCAM result side by side and save the figure
    """
    plt.subplot(1,2,1)
    plt.title('Image')
    plt.imshow(img)
    plt.tick_params(left=False,right=False,labelleft=False,labelbottom=False,bottom=False)

    plt.subplot(1,2,2)
    plt.title('Activations')
    plt.imshow(act,cmap='plasma')
    plt.tick_params(left=False,right=False,labelleft=False,labelbottom=False,bottom=False)

    plt.savefig(save_path)
    plt.show()
