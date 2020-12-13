import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

from pprint import pprint

from hidt.networks.enhancement.RRDBNet_arch import RRDBNet
from hidt.style_transformer import StyleTransformer
from hidt.utils.preprocessing import GridCrop, enhancement_preprocessing


def timegan_predict(img_path, time):
    config_path = './hidt/configs/daytime.yaml'
    gen_weights_path = './hidt/trained_models/generator/daytime.pt'
    device = 'cpu:0'

    style_transformer = StyleTransformer(
        config_path,
        gen_weights_path,
        inference_size=256, # output image size
        device=device
    )

    img = Image.open(img_path)


    with open('./hidt/styles.txt') as f:
        styles = f.read()

    styles = {style.split(',')[0]: torch.tensor([float(el) for el in style.split(',')[1][1:-1].split(' ')]) for style in styles.split('\n')[:-1]}

    style_to_transfer = styles[time]

    style_to_transfer = style_to_transfer.view(1, 1, 3, 1).to(device)

    with torch.no_grad():
        content_decomposition = style_transformer.get_content(img)[0]

        decoder_input = {
            'content': content_decomposition['content'],
            'intermediate_outputs': content_decomposition['intermediate_outputs'],
            'style': style_to_transfer
        }

        transferred = style_transformer.trainer.gen.decode(decoder_input)['images']
    
    
    output = (transferred[0].cpu().clamp(-1, 1).numpy().transpose(1, 2, 0) + 1.) / 2.

    print(output.shape)
    print("Success!! Success!! Success!! Success!!")

    # output = (output * 255).astype(np.uint8)
    frame_normed = 255 * (output - output.min()) / (output.max() - output.min())
    output = np.array(frame_normed, np.int)

    cv2.imwrite('./img/output.png', output)