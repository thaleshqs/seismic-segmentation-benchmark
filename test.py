import os
import torch
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from math import ceil
from PIL import Image

import core
from core.loader.data_loader import *
from core.models import load_empty_model


def generate_test_split(args: ArgumentParser, dataset: SeismicDataset) -> list:
    if args.first_slice < 0 or args.first_slice > len(dataset) - 1:
        raise IndexError(f'Index {args.first_slice} out of bounds for dataset of size {len(dataset)}')
    elif args.last_slice < 0 or args.last_slice > len(dataset) - 1:
        raise IndexError(f'Index {args.last_slice} out of bounds for dataset of size {len(dataset)}')
    else:
        return list(range(args.first_slice, args.last_slice + 1))


def save_image(output: torch.tensor, slice_idx: int, outputs_path: str) -> None:
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

    # Finding the class with the highest probability for each pixel
    class_indices = np.argmax(output, axis=0)

    # Mapping the class indices to colors
    image = color_map[class_indices]
    image = image.reshape((height, width, 3))

    image = Image.fromarray(image)
    image = image.rotate(270, expand=True)

    file_path = os.path.join(outputs_path, f'pred_{slice_idx}.png')
    image.save(file_path)


def store_outputs(args: ArgumentParser, outputs: list) -> None:
    # Creating the outputs folder if it does not exist
    if not os.path.exists(args.outputs_path):
        os.makedirs(args.outputs_path)
    
    results_folder = os.path.join(args.outputs_path, datetime.now().isoformat('#'))
    os.makedirs(results_folder)

    # Storing metadata
    with open(os.path.join(results_folder, f'metadata.json'), 'w') as json_buffer:
        json.dump(vars(args), json_buffer, indent=4)
    
    print('\nGenerating images from predictions...')
    idx = args.first_slice

    for batch in outputs:
        for image in batch:
            save_image(image.cpu(), slice_idx=idx, outputs_path=results_folder)
            idx += 1
    
    print(f'\nModel outputs saved as images in {results_folder}')


def run(args: ArgumentParser) -> list:
    if args.data_path.endswith('/'):
        args.data_path = args.data_path[:-1]
    
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f'Folder {args.data_path} does not exist.')

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f'No such file or directory for stored model: {args.model_path}')
    
    print('')
    print('Data path:        ', args.data_path)
    print('Stored model path:', args.model_path)
    print('')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Weighted loss ENABLED' if args.weighted_loss else 'Weighted loss DISABLED')
    print('Loading dataset...\n')

    dataset = SeismicDataset(
        data_path=args.data_path,
        orientation=args.orientation,
        compute_weights=args.weighted_loss
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
    test_indices = generate_test_split(args, dataset)

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
    print(f'Number of test examples: {len(test_indices)} (from range {args.first_slice} to {args.last_slice})')

    model = load_empty_model(args.architecture, dataset.n_classes)
    model = model.to(device)

    # Loading previously stored weights
    model.load_state_dict(torch.load(args.model_path))

    all_outputs = []

    with torch.no_grad():
        model.eval()
        test_loss = 0

        print(datetime.now().strftime('\n%Y/%m/%d %H:%M:%S'))
        print('Testing the model...\n')

        for images, labels in tqdm(test_loader, ascii=' >='):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            outputs = model(images)
            all_outputs.append(outputs)

            # Computing the loss
            loss = criterion(images=outputs, targets=labels.long())
            test_loss += loss.item()
        
        test_loss = test_loss / ceil((len(test_loader) / args.batch_size))

        print(f'Test loss: {test_loss}')
    
    return all_outputs


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
    parser.add_argument('-p', '--data-path',
        dest='data_path',
        type=str,
        help='Path to the folder containing the dataset and its labels in .npy format'
    )
    parser.add_argument('-b', '--batch-size',
        dest='batch_size',
        type=int,
        default=16,
        help='Batch Size'
    )
    parser.add_argument('-d', '--device',
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
    parser.add_argument('-i', '--first-slice',
        dest='first_slice',
        type=int,
        default=0,
        help='Index of the first slice from the test set'
    )
    parser.add_argument('-j', '--last-slice',
        dest='last_slice',
        type=int,
        help='Index of the last slice from the test set'
    )
    parser.add_argument('-s', '--store-outputs',
        dest='store_outputs',
        action='store_true',
        default=False,
        help='Whether to store the model outputs as images'
    )
    parser.add_argument('-r', '--outputs-path',
        dest='outputs_path',
        type=str,
        default=os.path.join(os.getcwd(), 'outputs'),
        help='Directory for storing model outputs'
    )

    args = parser.parse_args(args=None)
    outputs = run(args)

    if args.store_outputs:
        store_outputs(args, outputs)
