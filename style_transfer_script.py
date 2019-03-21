from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models


# get 'features' portion of VGG19
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

# move the model to GPU, if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)

def load_image(img_path, max_size=400, shape=None):
    '''
    Load in and transform an image, making sure image is <= 400 pixels in the
    x-y dims
    '''
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([transforms.Resize(size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

# load in content and image design
content_image = load_image('images/content.jpg').to(device)
style_image = load_image('images/style.jpeg', shape=content_image.shape[-2:]).to(device)

def im_convert(tensor):
    '''
    Helper function for un-normalizing an image
    and converting it from Tensor to numpy for display that image
    '''
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225))\
           + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

#display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content_image))
ax2.imshow(im_convert(style_image))
print('Draw successfully')
# plt.show()

def get_features(image, model, layers=None):
    '''Run an image through a model and get the features for a set of layer.
    Default layers are for VGG net matching Gatys
    '''
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
    features = {}
    tmp_image = image
    for name, layer in model._modules.items():
        tmp_image = layer(tmp_image)
        if name in layers:
            features[layers[name]] = tmp_image

    return features

def gram_matrix(tensor):
    '''
    Calculate gram matrix for a given tensor
    '''
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())

    return gram

content_features = get_features(content_image, vgg)
style_features = get_features(style_image, vgg)
style_grams = {layer: gram_matrix(style_features[layer])
              for layer in style_features}

# create a target by copying content image
target = content_image.clone().requires_grad_(True).to(device)

show_every = 100
optimizer = optim.Adam([target], lr=0.003)
steps = 2000

style_weights = {
    'conv1_1': 0.1,
    'conv2_1': 0.2,
    'conv3_1': 0.4,
    'conv4_1': 0.6,
    'conv4_2': 0.6,
    'conv5_1': 0.7
}

content_weight = 1
style_weight = 0.6

for i in range(1, steps + 1):
    print('Running Epoch', i)
    target_features = get_features(target, vgg)

    # content loss
    content_loss = torch.mean((target_features['conv4_2']
                               - content_features['conv4_2']) ** 2)

    # style loss
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] \
                           * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (d * h * w)

    # total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % show_every == 0:
        print('Total loss', total_loss.item())
        numpy_image = im_convert(target)
        plt.imshow(numpy_image)
        plt.savefig(f'image{i}.jpg')
        # plt.show()
        # im = Image.fromarray(numpy_image)
        # im.save(f'image{i}.jpg')
