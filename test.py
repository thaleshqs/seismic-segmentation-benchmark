import os
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from math import ceil
from PIL import Image
from sklearn.model_selection import KFold
from torchvision.transforms import ToPILImage

import core
from core.loader.data_loader import *
from core.models import load_empty_model


def train_test_split(dataset: SeismicDataset) -> list:
    kf = KFold(n_splits=5, shuffle=False)
    
    return [(train_idx.tolist(), test_idx.tolist()) for train_idx, test_idx in kf.split(dataset)]


def tensor_to_image(output: torch.tensor, is_label=False) -> Image:
    # Viridis color palette
    color_map = np.array([
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
    ], dtype=np.uint8)
    
    _, height, width = output.shape

    # For the case that it is an image
    if is_label:
        output = output.numpy().astype(np.uint8)
    # For the case that it is a prediction
    else:
        # Finding the class with the highest probability for each pixel
        output = np.argmax(output, axis=0)

    # Mapping the class indices to colors
    image = color_map[output]
    image = image.reshape((height, width, 3))

    image = Image.fromarray(image)
    image = image.rotate(270, expand=True)

    return image


def save_images(args: ArgumentParser, outputs: list) -> None:
    # Creating the outputs folder if it does not exist
    if not os.path.exists(args.outputs_path):
        os.makedirs(args.outputs_path)
    
    results_folder = os.path.join(args.outputs_path, datetime.now().isoformat('#'))

    os.makedirs(results_folder)
    os.makedirs(os.path.join(results_folder, 'preds'))
    os.makedirs(os.path.join(results_folder, 'labels'))

    # Storing metadata
    with open(os.path.join(results_folder, f'metadata.json'), 'w') as json_buffer:
        json.dump(vars(args), json_buffer, indent=4)
    
    print('\nGenerating images from predictions...')

    for idx, (pred, label) in outputs.items():
        pred = tensor_to_image(pred.cpu())
        label = tensor_to_image(label.cpu(), is_label=True)

        pred.save(os.path.join(results_folder, f'preds/pred_{idx}.png'))
        label.save(os.path.join(results_folder, f'labels/label_{idx}.png'))

    print(f'\nModel outputs saved as images in {results_folder}')


def run(args: ArgumentParser) -> dict:    
    print('')
    print('Data path:        ', args.data_path)
    print('Labels path:      ', args.labels_path)
    print('Stored model path:', args.model_path)
    print('')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Weighted loss ENABLED' if args.weighted_loss else 'Weighted loss DISABLED')
    print('Loading dataset...\n')

    dataset = SeismicDataset(
        data_path=args.data_path,
        labels_path=args.labels_path,
        orientation=args.orientation,
        compute_weights=args.weighted_loss,
        faulty_slices_list=args.faulty_slices_list
    )

    if args.weighted_loss:
        class_weights = torch.tensor(dataset.get_class_weights(), device=device, requires_grad=False)
        class_weights = class_weights.float()
    else:
        class_weights = None
    
    loss_map = {
        'cel': ('CrossEntropyLoss', {'reduction': 'sum', 'weight': class_weights})
    }

    # Defining the loss function
    loss_name, loss_args = loss_map[args.loss_function]
    criterion = getattr(core.loss, loss_name)(**loss_args)

    print(f'Testing with {"INLINES" if args.orientation == "in" else "CROSSLINES"}')

    # Retrieving only a subset of slices for testing
    _, test_indices = train_test_split(dataset)[args.fold_number - 1]

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SequentialSampler(test_indices),
    )

    print('\nTesting model...')
    print('Architecture:   ', args.architecture.upper())
    print('Loss function:  ', args.loss_function)
    print('Device:         ', device)
    print('Batch size:     ', args.batch_size)

    print('')
    print(f'Number of test examples: {len(test_indices)} (slices {test_indices[0]} to {test_indices[-1]})')

    model = load_empty_model(args.architecture, dataset.n_classes)
    model = model.to(device)

    # Loading previously stored weights
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f'No such file or directory for stored model: {args.model_path}')
    
    model.load_state_dict(torch.load(args.model_path))

    results = {}

    with torch.no_grad():
        model.eval()
        test_loss = 0

        print(datetime.now().strftime('\n%Y/%m/%d %H:%M:%S'))
        print('Testing the model...\n')
        
        slice_idx = test_indices[0]

        for images, labels in tqdm(test_loader, ascii=' >='):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            
            print(torch.max(labels))

            outputs = model(images)

            # Iterating over the batch
            for pred, label in zip(outputs, labels):
                results[slice_idx] = (pred, label)
                slice_idx += 1

            # Computing the loss
            loss = criterion(images=outputs, targets=labels.long())
            test_loss += loss.item()
        
        test_loss = test_loss / ceil((len(test_loader) / args.batch_size))

        print(f'Test loss: {test_loss}')
    
    return results


if __name__ == '__main__':
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('-m', '--model-path',
        dest='model_path',
        type=str,
        help='Directory for loading saved model'
    )
    parser.add_argument('-a', '--architecture',
        dest='architecture',
        type=str,
        help='Architecture to use [segnet, unet, deconvnet]',
        choices=['segnet', 'unet', 'deconvnet']
    )
    parser.add_argument('-d', '--data-path',
        dest='data_path',
        type=str,
        help='Path to the data file in numpy or segy format'
    )
    parser.add_argument('-l', '--labels-path',
        dest='labels_path',
        type=str,
        help='Path to the labels file in numpy format'
    )
    parser.add_argument('-b', '--batch-size',
        dest='batch_size',
        type=int,
        default=16,
        help='Batch Size'
    )
    parser.add_argument('-D', '--device',
        dest='device',
        type=str,
        default='cuda:0',
        help='Device to train on [cuda:n]'
    )
    parser.add_argument('-L', '--loss-function',
        dest='loss_function',
        type=str,
        default='cel',
        help='Loss function to use [cel (Cross_Entropy Loss)]',
        choices=['cel']
    )
    parser.add_argument('-W', '--weighted-loss',
        dest='weighted_loss',
        action='store_true',
        default=False,
        help='Whether to use class weights in the loss function'
    )
    parser.add_argument('-O', '--orientation',
        dest='orientation',
        type=str,
        default='in',
        help='Whether the model should be trained using inlines or crosslines',
        choices=['in', 'cross']
    )
    parser.add_argument('-n', '--fold-number',
        dest='fold_number',
        type=int,
        default=1,
        help='Fold number (from 1 to 5) of the data used for testing during training',
        choices=[1, 2, 3, 4, 5]
    )
    parser.add_argument('-f', '--faulty-slices-list',
        dest='faulty_slices_list',
        type=str,
        default=None,
        help='Path to a json file containing a list of faulty slices to remove'
    )
    parser.add_argument('-s', '--store-outputs',
        dest='store_outputs',
        action='store_true',
        default=False,
        help='Whether to store the model outputs as images'
    )
    parser.add_argument('-p', '--outputs-path',
        dest='outputs_path',
        type=str,
        default=os.path.join(os.getcwd(), 'outputs'),
        help='Directory for storing model outputs'
    )

    args = parser.parse_args(args=None)
    outputs = run(args)

    if args.store_outputs:
        save_images(args, outputs)
