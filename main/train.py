"""Train the model"""
import argparse
import logging
import os
import os.path
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='all', help="The Model Name")
parser.add_argument('--md', default='experiments', help="The Model Directory")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")

def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader), ascii=True) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.float().cuda(), labels_batch.float().cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item() # loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def train_and_evaluate(model, optimizer, scheduler, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_loss = 1e10
    best_test_loss = 1e10

    for epoch in range(params.num_epochs):
        logging.info("Generate the train and test datasets...")
        # fetch dataloaders for every epoch
        dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], args.data_dir, params)
        logging.info("- done.")

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, dataloaders['train'], metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, dataloaders['val'], metrics, params, 'Val')
        val_loss = val_metrics['rmse']
        is_best_val = val_loss<=best_val_loss

        # Evaluate for one epoch on test set
        test_metrics = evaluate(model, loss_fn, dataloaders['test'], metrics, params, 'Test')
        test_loss = test_metrics['rmse']
        is_best_test = test_loss<=best_test_loss

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best_val,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best_val:
            logging.info("- Found new best val result")
            best_val_loss = val_loss
            # Save metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        if is_best_test:
            logging.info("- Found new best test result")
            best_test_loss = test_loss
            # Save metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_test_best_weights.json")
            utils.save_dict_to_json(test_metrics, best_json_path)

def run(args=None):
    # Load the parameters from json file
    args.model_dir = args.md + '/' + args.model
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    print(os.path.join(args.model_dir, '_train.log'))
    utils.set_logger(os.path.join(args.model_dir, '_train.log'))

    # Define the model and optimizer
    model = getattr(net, args.model)(params).cuda() if params.cuda else getattr(net, args.model)(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=10e-5)
    # optimizer = optim.RSMprop(model.parameters(), lr=params.learning_rate)
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = None
    
    # fetch loss function and metrics
    loss_fn = nn.MSELoss()
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, optimizer, scheduler, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.model == 'all':
        MODELS = [f for f in os.listdir(args.md) if not os.path.isfile(os.path.join(args.md, f))]
        for m in MODELS:
            args.model = m
            run(args)
    else:
        run(args)