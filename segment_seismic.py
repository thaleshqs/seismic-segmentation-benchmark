import torch
import json
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from math import ceil
from sklearn.model_selection import KFold

import core
from core.loader.data_loader import *
from core.models import load_empty_model
from core.metrics import RunningMetrics, EarlyStopper


def train_test_split(args: ArgumentParser, dataset: SeismicDataset) -> list:
    if args.cross_validation:
        kf = KFold(n_splits=5, shuffle=False)

        return list(kf.split(dataset))
    else:
        test_size  = int(len(dataset)*args.test_ratio)

        return [(
            list(range(test_size, len(dataset))),
            list(range(0, test_size))
        )]


def get_model_name(args: ArgumentParser, dataset_name: str) -> str:
    model_name = ''

    model_name += 'data_'    + dataset_name + '_'
    model_name += 'arch_'    + args.architecture.upper() + '_'
    model_name += 'ori_'     + args.orientation.upper() + '_'
    model_name += 'loss_'    + args.loss_function.upper() + '_'
    model_name += 'opt_'     + args.optimizer + '_'
    model_name += 'batch_'   + str(args.batch_size) + '_'
    model_name += 'epochs_'  + str(args.n_epochs) + '_'
    model_name += 'weights_' + str(args.weighted_loss) + '_'
    model_name += 'cross_'   + str(args.cross_validation) + '_'
    model_name += datetime.now().isoformat('#')

    return model_name


def store_results(args: ArgumentParser, dataset_name: str, results: dict) -> None:
    # Creating the results folder if it does not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    model_name = get_model_name(args, dataset_name)

    results_dir = os.path.join(args.results_path, model_name)
    os.makedirs(results_dir)

    for fold_number in sorted(results.keys()):
        model  = results[fold_number]['model']
        scores = {key: value for key, value in results[fold_number].items() if key != 'model'}

        torch.save(model.state_dict(), os.path.join(results_dir, f'model_fold_{fold_number + 1}.pt'))
        
        with open(os.path.join(results_dir, f'scores_fold_{fold_number + 1}.json'), 'w') as json_buffer:
            json.dump(scores, json_buffer, indent=4)


def run(args: ArgumentParser) -> dict:
    if args.data_path.endswith('/'):
        args.data_path = args.data_path[:-1]

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f'Folder {args.data_path} does not exist.')
    
    print('')
    print('Data path:    ', args.data_path)
    print('Results path: ', args.results_path)
    print('')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Weighted loss ENABLED' if args.weighted_loss else 'Weighted loss DISABLED')
    print('Loading dataset...\n')

    dataset = SeismicDataset(
        data_path=args.data_path,
        orientation=args.orientation,
        compute_weights=args.weighted_loss
    )

    # Weights are inversely proportional to the frequency of the classes in the training set
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

    print(f'Training with {"inlines" if args.orientation == "in" else "crosslines"}')

    # Splitting the data into train and test
    splits  = train_test_split(args, dataset)
    results = {}

    for fold_number, (train_indices, test_indices) in enumerate(splits):
        if args.cross_validation:
            print(f'\n======== FOLD {fold_number + 1}/5 ========')

        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        )

        print('\nCreating model...')
        print('Architecture:   ', args.architecture.upper())
        print('Optimizer:      ', args.optimizer)
        print('Loss function:  ', args.loss_function)
        print('Learning rate:  ', args.learning_rate)
        print('Device:         ', device)
        print('Batch size:     ', args.batch_size)
        print('N. of epochs:   ', args.n_epochs)

        print('')
        print('Number of training examples:', len(train_indices))
        print('Number of test examples:    ', len(test_indices))
        
        model = load_empty_model(args.architecture, dataset.get_n_classes())
        model = model.to(device)

        # Defining the optimizer
        optimizer_map = {
            'adam': torch.optim.Adam,
            'sgd' : torch.optim.SGD
        }

        OptimizerClass = optimizer_map[args.optimizer]
        optimizer      = OptimizerClass(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Initializing metrics
        train_metrics = RunningMetrics(n_classes=dataset.get_n_classes(), bf1_threshold=2)
        test_metrics  = RunningMetrics(n_classes=dataset.get_n_classes(), bf1_threshold=2)

        early_stopper = EarlyStopper(patience=3, min_delta=10)

        # Storing the loss for each epoch
        train_loss_list = []
        test_loss_list  = []

        for epoch in range(args.n_epochs):
            # Training phase
            model.train()
            train_loss = 0

            print(datetime.now().strftime('\n%Y/%m/%d %H:%M:%S'))
            print(f'Training on epoch {epoch + 1}/{args.n_epochs}\n')

            for images, labels in tqdm(train_loader, ascii=' >='):
                optimizer.zero_grad()

                images = images.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)

                outputs = model(images)

                # Updating the running metrics
                train_metrics.update(images=outputs, targets=labels)

                # Computing the loss and updating weights
                loss = criterion(images=outputs, targets=labels.long())
                train_loss += loss.item()
                loss.backward()

                optimizer.step()
            
            train_loss = train_loss / ceil((len(train_loader) / args.batch_size))
            train_scores = train_metrics.get_scores()

            print(f'Train loss: {train_loss}')
            print(f'Train mIoU: {train_scores["mean_iou"]}')

            train_loss_list.append(train_loss)
            train_metrics.reset()
        
        # Testing phase
        with torch.no_grad():
            model.eval()
            test_loss = 0

            print(datetime.now().strftime('\n%Y/%m/%d %H:%M:%S'))
            print('Testing the model...\n')

            for images, labels in tqdm(test_loader, ascii=' >='):
                images = images.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)

                outputs = model(images)

                # Updating the running metrics
                test_metrics.update(images=outputs, targets=labels)

                # Computing the loss
                loss = criterion(images=outputs, targets=labels.long())
                test_loss += loss.item()
            
            test_loss = test_loss / ceil((len(test_loader) / args.batch_size))
            test_scores = test_metrics.get_scores()

            print(f'Test loss: {test_loss}')
            print(f'Test mIoU: {test_scores["mean_iou"]}')

            test_loss_list.append(test_loss)
            test_metrics.reset()
            
        # if early_stopper.early_stop(test_loss):             
        #     break
        
        results[fold_number] = {
            'model'        : model,
            'train_scores' : train_scores,
            'test_scores'  : test_scores,
            'train_losses' : train_loss_list,
            'test_losses'  : test_loss_list,
            'train_indices': train_indices,
            'test_indices' : test_indices
        }
    
    return results


if __name__ == '__main__':
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('-a', '--architecture',
        dest='architecture',
        type=str,
        help='Architecture to use [segnet, unet, deconvnet]',
        choices=['segnet', 'unet', 'deconvnet']
    )
    parser.add_argument('-dp', '--data-path',
        dest='data_path',
        type=str,
        help='Path to the folder containing the dataset and its labels in .npy format'
    )
    parser.add_argument('-bs', '--batch-size',
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
    parser.add_argument('-cv', '--cross-validation',
        dest='cross_validation',
        action='store_true',
        default=False,
        help='Whether to use 5-fold cross validation'
    )
    parser.add_argument('-l', '--loss-function',
        dest='loss_function',
        type=str,
        default='cel',
        help='Loss function to use [cel (Cross_Entropy Loss)]',
        choices=['cel']
    )
    parser.add_argument('-op', '--optimizer',
        dest='optimizer',
        type=str,
        default='adam',
        help='Optimizer to use [adam, sgd (Stochastic Gradient Descent)]',
        choices=['adam', 'sgd']
    )
    parser.add_argument('-lr', '--learning-rate',
        dest='learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument('-wd', '--weight-decay',
        dest='weight_decay',
        type=float,
        default=1e-5,
        help='L2 regularization. Value 0 indicates no weight decay'
    )
    parser.add_argument('-wl', '--weighted-loss',
        dest='weighted_loss',
        action='store_true',
        default=False,
        help='Whether to use class weights in the loss function'
    )
    parser.add_argument('-e', '--n-epochs',
        dest='n_epochs',
        type=int,
        default=50,
        help='Number of epochs'
    )
    parser.add_argument('-o', '--orientation',
        dest='orientation',
        type=str,
        default='in',
        help='Whether the model should be trained using inlines or crosslines',
        choices=['in', 'cross']
    )
    parser.add_argument('-r', '--test-ratio',
        dest='test_ratio',
        type=float,
        default=0.2,
        help='Percentage of the data used for the test set'
    )
    parser.add_argument('-s', '--store-results',
        dest='store_results',
        action='store_true',
        default=True,
        help='Whether to store the model weights and metrics'
    )
    parser.add_argument('-rp', '--results-path',
        dest='results_path',
        type=str,
        default=os.path.join(os.getcwd(), 'results'),
        help='Directory for storing execution results'
    )

    args = parser.parse_args(args=None)
    results = run(args)

    dataset_name = os.path.basename(args.data_path).rsplit('.')[0]

    if args.store_results:
        store_results(args, dataset_name, results)

    # # Loading a model if it was previously stored
    # if args.stored_model_path:
    #     if os.path.isfile(args.stored_model_path):
    #         print(f'Loading stored model {args.stored_model_path}')

    #         model = load_empty_model(args.architecture, 1, dataset.n_classes)
    #         model.load_state_dict(torch.load(args.stored_model_path))
    #     else:
    #         raise FileNotFoundError(f'No such file or directory for stored model: {args.stored_model_path}')
    # else:
    #     model = train(train_set, device=device, criterion=criterion, args=args)

    # test(test_set, model=model, device=device, criterion=criterion, args=args)
