import os,argparse,time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torch.utils.data.distributed

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from dataloaders.deep_moji import DeepMojiDataset
from networks.deepmoji_sa import DeepMojiModel
from networks.discriminator import Discriminator


from tqdm import tqdm, trange
from networks.customized_loss import DiffLoss

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.eval_metrices import group_evaluation, leakage_evaluation

from pathlib import Path, PureWindowsPath

import argparse

# train a discriminator 1 epoch
def adv_train_epoch(model, discriminators, iterator, adv_optimizers, criterion, device, args):
    """"
    Train the discriminator to get a meaningful gradient
    """

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    for discriminator in discriminators:
        discriminator.train()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    
    for batch in iterator:
        
        text = batch[0]
        tags = batch[1].long()
        p_tags = batch[2].long()

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        hs = model.hidden(text).detach()
        
        # iterate all discriminators
        for discriminator, adv_optimizer in zip(discriminators, adv_optimizers):
        
            adv_optimizer.zero_grad()

            adv_predictions = discriminator(hs)

        
            loss = criterion(adv_predictions, p_tags)

            # encrouge orthogonality
            if args.DL == True:
                # Get hidden representation.
                adv_hs_current = discriminator.hidden_representation(hs)
                for discriminator2 in discriminators:
                    if discriminator != discriminator2:
                        adv_hs = discriminator2.hidden_representation(hs)
                        # Calculate diff_loss
                        # should not include the current model
                        difference_loss = args.diff_LAMBDA * args.diff_loss(adv_hs_current, adv_hs)
                        loss = loss + difference_loss
                        
            loss.backward()
        
            adv_optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# evaluate the discriminator
def adv_eval_epoch(model, discriminators, iterator, criterion, device, args):

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    for discriminator in discriminators:
        discriminator.eval()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    

    preds = {i:[] for i in range(args.n_discriminator)}
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).long()
        
        # extract hidden state from the main model
        hs = model.hidden(text)
        # let discriminator make predictions

        for index, discriminator in enumerate(discriminators):
            adv_pred = discriminator(hs)
        
            loss = criterion(adv_pred, p_tags)
                        
            epoch_loss += loss.item()
        
            adv_predictions = adv_pred.detach().cpu()
            preds[index] += list(torch.argmax(adv_predictions, axis=1).numpy())


        tags = tags.cpu().numpy()

        labels += list(tags)
        
        private_labels += list(batch[2].cpu().numpy())
        
    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)

# train the main model with adv loss
def train_epoch(model, discriminators, iterator, optimizer, criterion, device, args):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for discriminator in discriminators:
        discriminator.train()

    # activate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = True
    
    for batch in iterator:
        
        text = batch[0]
        tags = batch[1].long()
        p_tags = batch[2].float()

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        optimizer.zero_grad()
        # main model predictions
        predictions = model(text)
        # main tasks loss
        loss = criterion(predictions, tags)

        if args.adv:
            # discriminator predictions
            p_tags = p_tags.long()

            hs = model.hidden(text)

            for discriminator in discriminators:
                adv_predictions = discriminator(hs)
            
                loss = loss + (criterion(adv_predictions, p_tags) / len(discriminators))
                        
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# to evaluate the main model
def eval_main(model, iterator, criterion, device, args):
    
    epoch_loss = 0
    
    model.eval()
    
    preds = []
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]

        tags = batch[1]
        # tags = batch[2] #Reverse
        p_tags = batch[2]

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).float()

        predictions = model(text)
        
        loss = criterion(predictions, tags)
                        
        epoch_loss += loss.item()
        
        predictions = predictions.detach().cpu()
        tags = tags.cpu().numpy()

        preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())
    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)

def log_uniform(power_low, power_high):
    return np.power(10, np.random.uniform(power_low, power_high))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--hidden_size', type=int, default = 300)
    parser.add_argument('--emb_size', type=int, default = 2304)
    parser.add_argument('--num_classes', type=int, default = 2)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--adv_level', type=int, default = -1)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--starting_power', type=int)
    parser.add_argument('--LAMBDA', type=float, default=0.8)
    parser.add_argument('--n_discriminator', type=int, default = 0)
    parser.add_argument('--adv_units', type=int, default = 256)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--DL', action='store_true')
    parser.add_argument('--diff_LAMBDA', type=float, default=1000)
    parser.add_argument('--data_path', type=str)

    args = parser.parse_args()

    # file names
    experiment_type = "adv_Diverse"
    
    # path to checkpoints
    main_model_path = "models\\deepnoji_model_{}.pt".format(experiment_type)
    adv_model_path = "models\\discriminator_{}_{}.pt"
    
    # DataLoader Parameters
    params = {'batch_size': 512,
            'shuffle': True,
            'num_workers': 0}
    # Device
    device = torch.device("cuda")

    data_path = args.data_path
    # Load data
    train_data = DeepMojiDataset(args, data_path, "train", ratio=args.ratio, n = 100000)
    dev_data = DeepMojiDataset(args, data_path, "dev")
    test_data = DeepMojiDataset(args, data_path, "test")

    # Data loader
    training_generator = torch.utils.data.DataLoader(train_data, **params)
    validation_generator = torch.utils.data.DataLoader(dev_data, **params)
    test_generator = torch.utils.data.DataLoader(test_data, **params)

    # Init model
    model = DeepMojiModel(args)

    model = model.to(device)

    # Init discriminators
    # Number of discriminators
    n_discriminator = args.n_discriminator

    discriminators = [Discriminator(args, args.hidden_size, 2) for _ in range(n_discriminator)]
    discriminators = [dis.to(device) for dis in discriminators]

    diff_loss = DiffLoss()
    args.diff_loss = diff_loss

    # Init optimizers
    LEARNING_RATE = args.lr
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    adv_optimizers = [Adam(filter(lambda p: p.requires_grad, dis.parameters()), lr=1e-1*LEARNING_RATE) for dis in discriminators]

    # Init learing rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 2)

    # Init criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    
    best_loss, valid_preds, valid_labels, _ = eval_main(
                                                        model = model, 
                                                        iterator = validation_generator, 
                                                        criterion = criterion, 
                                                        device = device, 
                                                        args = args
                                                        )

    best_acc = accuracy_score(valid_labels, valid_preds)
    best_epoch = 60

    for i in trange(60):
        train_epoch(
                    model = model, 
                    discriminators = discriminators, 
                    iterator = training_generator, 
                    optimizer = optimizer, 
                    criterion = criterion, 
                    device = device, 
                    args = args
                    )

        valid_loss, valid_preds, valid_labels, _ = eval_main(
                                                            model = model, 
                                                            iterator = validation_generator, 
                                                            criterion = criterion, 
                                                            device = device, 
                                                            args = args
                                                            )
        valid_acc = accuracy_score(valid_preds, valid_labels)
        # learning rate scheduler
        scheduler.step(valid_loss)

        # early stopping
        if valid_loss < best_loss:
            if i >= 5:
                best_acc = valid_acc
                best_loss = valid_loss
                best_epoch = i
                torch.save(model.state_dict(), main_model_path)
        else:
            if best_epoch+5<=i:
                break

        # Train discriminator untile converged
        # evaluate discriminator 
        best_adv_loss, _, _, _ = adv_eval_epoch(
                                                model = model, 
                                                discriminators = discriminators, 
                                                iterator = validation_generator, 
                                                criterion = criterion, 
                                                device = device, 
                                                args = args
                                                )
        best_adv_epoch = -1
        for k in range(100):
            adv_train_epoch(
                            model = model, 
                            discriminators = discriminators, 
                            iterator = training_generator, 
                            adv_optimizers = adv_optimizers, 
                            criterion = criterion, 
                            device = device, 
                            args = args
                            )
            adv_valid_loss, _, _, _ = adv_eval_epoch(
                                                    model = model, 
                                                    discriminators = discriminators, 
                                                    iterator = validation_generator, 
                                                    criterion = criterion, 
                                                    device = device, 
                                                    args = args
                                                    )
                
            if adv_valid_loss < best_adv_loss:
                    best_adv_loss = adv_valid_loss
                    best_adv_epoch = k
                    for j in range(args.n_discriminator):
                        torch.save(discriminators[j].state_dict(), adv_model_path.format(experiment_type, j))
            else:
                if best_adv_epoch + 5 <= k:
                    break
        for j in range(args.n_discriminator):
            discriminators[j].load_state_dict(torch.load(adv_model_path.format(experiment_type, j)))

    model.load_state_dict(torch.load(main_model_path))
    
    # Evaluation
    test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
    preds = np.array(preds)
    labels = np.array(labels)
    p_labels = np.array(p_labels)
    
    eval_metrices = group_evaluation(preds, labels, p_labels, silence=False)
    
    print("Overall Accuracy", (eval_metrices["Accuracy_0"]+eval_metrices["Accuracy_1"])/2)