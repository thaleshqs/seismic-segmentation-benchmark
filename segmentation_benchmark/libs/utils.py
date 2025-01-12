# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/libs/utils.ipynb.

# %% auto 0
__all__ = ['tensor_to_image', 'save_images', 'store_results']

# %% ../../nbs/libs/utils.ipynb 2
import os
import json
import torch
import numpy as np
from datetime import datetime
from PIL import Image


def tensor_to_image(output, color_palette='viridis'):
    # Defining color palettes
    color_map = {
        'viridis': np.array([
            (253, 231, 37),
            (181, 222, 43),
            (110, 206, 88),
            (53, 183, 121),
            (31, 158, 137),
            (38, 130, 142),
            (49, 104, 142),
            (62, 73, 137),
            (72, 40, 120),
            (68, 1, 84)
        ], dtype=np.uint8),

        'binary': np.array([
            (0, 0, 0),
            (255, 255, 255)
        ], dtype=np.uint8)
    }
    
    _, height, width = output.shape

    # Finding the class with the highest probability for each pixel
    output = np.argmax(output, axis=0)

    # Mapping the class indices to colors
    image = color_map[color_palette][output]
    image = image.reshape((height, width, 3))

    image = Image.fromarray(image)
    image = image.rotate(270, expand=True)

    return image


def save_images(preds, preds_path, color_palette):    
    for idx, pred in preds.items():
        pred = tensor_to_image(pred.cpu(), color_palette=color_palette)
        pred.save(os.path.join(preds_path, f'pred_{idx}.png'))


def store_results(args, results, n_classes):
    # Creating the results folder if it does not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    results_folder = os.path.join(args.results_path, datetime.now().isoformat('#'))
    os.makedirs(results_folder)

    images_folder = os.path.join(results_folder, 'preds')
    os.makedirs(images_folder)

    # Storing metadata
    with open(os.path.join(results_folder, f'metadata.json'), 'w') as json_buffer:
        json.dump(vars(args), json_buffer, indent=4)
    
    for fold_number in range(len(results)):
        suffix = f'_fold_{fold_number + 1}' if args.cross_validation else ''

        model = results[fold_number]['model']
        scores = {
            key: value for key, value in results[fold_number].items() if key not in ['model', 'preds']
        }

        # Storing model weights
        if args.train:
            torch.save(model.state_dict(), os.path.join(results_folder, 'model' + suffix + '.pt'))

        # Storing metric scores
        with open(os.path.join(results_folder, 'scores' + suffix + '.json'), 'w') as json_buffer:
            json.dump(scores, json_buffer, indent=4)
        
        # Storing model outputs as images
        color_palette = 'binary' if n_classes == 2 else 'viridis'
        save_images(results[fold_number]['preds'], images_folder, color_palette)
    
    print(f'\nResults saved in {args.results_path}')
