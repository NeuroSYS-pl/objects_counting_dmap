"""This script apply a chosen model on a given image.

One needs to choose a network architecture and provide the corresponding
state dictionary.

Example:

    $ python infer.py -n UNet -c mall_UNet.pth -i seq_000001.jpg

The script also allows to visualize the results by drawing a resulting
density map on the input image.

Example:

    $ $ python infer.py -n UNet -c mall_UNet.pth -i seq_000001.jpg --visualize

"""
import os

import click
import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from PIL import Image

from model import UNet, FCRN_A


@click.command()
@click.option('-i', '--image_path',
              type=click.File('r'),
              required=True,
              help="A path to an input image.")
@click.option('-n', '--network_architecture',
              type=click.Choice(['UNet', 'FCRN_A']),
              required=True,
              help='Model architecture.')
@click.option('-c', '--checkpoint',
              type=click.File('r'),
              required=True,
              help='A path to a checkpoint with weights.')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('--one_channel',
              is_flag=True,
              help="Turn this on for one channel images (required for ucsd).")
@click.option('--pad',
              is_flag=True,
              help="Turn on padding for input image (required for ucsd).")
@click.option('--visualize',
              is_flag=True,
              help="Visualize predicted density map.")
def infer(image_path: str,
          network_architecture: str,
          checkpoint: str,
          unet_filters: int,
          convolutions: int,
          one_channel: bool,
          pad: bool,
          visualize: bool):
    """Run inference for a single image."""
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if one_channel else 3

    # initialize a model based on chosen network_architecture
    network = {
        'UNet': UNet,
        'FCRN_A': FCRN_A
    }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions).to(device)

    # load provided state dictionary
    # note: by default train.py saves the model in data parallel mode
    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(checkpoint.name))
    network.eval()

    img = Image.open(image_path.name)

    # padding was applied for ucsd images to allow down and upsampling
    if pad:
        img = Image.fromarray(np.pad(img, 1, 'constant', constant_values=0))

    # network's output represents a density map
    density_map = network(TF.to_tensor(img).unsqueeze_(0))

    # note: density maps were normalized to 100 * no. of objects
    n_objects = torch.sum(density_map).item() / 100

    print(f"The number of objects found: {n_objects}")

    if visualize:
        _visualize(img, density_map.squeeze().cpu().detach().numpy())


def _visualize(img, dmap):
    """Draw a density map onto the image."""
    # keep the same aspect ratio as an input image
    fig, ax = plt.subplots(figsize=figaspect(1.0 * img.size[1] / img.size[0]))
    fig.subplots_adjust(0, 0, 1, 1)

    # plot a density map without axis
    ax.imshow(dmap, cmap="hot")
    plt.axis('off')
    fig.canvas.draw()

    # create a PIL image from a matplotlib figure
    dmap = Image.frombytes('RGB',
                           fig.canvas.get_width_height(),
                           fig.canvas.tostring_rgb())

    # add a alpha channel proportional to a density map value
    dmap.putalpha(dmap.convert('L'))

    # display an image with density map put on top of it
    Image.alpha_composite(img.convert('RGBA'), dmap.resize(img.size)).show()


if __name__ == "__main__":
    infer()
