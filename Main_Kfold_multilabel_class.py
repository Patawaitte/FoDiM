from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt
import random
import pandas as pd
from glob import glob
import torch
import torch.nn as nn
import numpy as np
import math
from utils.dataset_multi_label_class import MultiDisDataset as myDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
sns.cubehelix_palette(as_cmap=True)

from models.TempCNN_multi import TempCNN
from models.TransformerModel_relative_multi import TransformerModel

from scipy.ndimage import gaussian_filter1d


print(torch.cuda.is_available())

cuda_device = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
learning_rate = 0.001
num_epochs = 60
batch_size = 32


def criterion_guauss_w(loss_func,outputs,data, sigma):
    """
    Custum weighted loss fonction.
    A gaussian filter can be apply on the date prediction

    Args:
        loss_func : Loss function to apply.
        outputs : The output of the model.
        data: True multi label.
        sigma : sigma of the gaussian_filter1d function.


    Returns:
        The loss.
    """

    losses = 0

    for i, key in enumerate(outputs):
        if key=='date':
            # Count number of unique label
            samples = [64081,479,213,356,753,800,881,8839,543,459,617,4504,1537,539,816,586,970,596,1204,526,1119,1000,1459,5072,1030,1733,2412,3700,5677,3114,2551,2456,3882,2747,5393,2637,768,31]

            # Gaussian weight filter
            #yh =data['labels'][f'label_{key}']
            #yg=gaussian_filter1d(yh, sigma=sigma)
            #print('yd', yg[0])
            #ygt = torch.tensor(yg)

            ygt=data['labels'][f'label_{key}']
            weight_tensor =  torch.FloatTensor(1.0 - (samples/ np.sum(samples))).to(device)
            loss = loss_func(outputs[key], ygt.to(device))
            loss = (loss * weight_tensor).mean()


        elif key=='type':
            # Count number of unique label
            samples = [14622,	17571,	41456,	28398,	23126,	7579,	3320]

            ygt=data['labels'][f'label_{key}']
            weight_tensor =  torch.FloatTensor(1.0 - (samples/ np.sum(samples))).to(device)
            loss = loss_func(outputs[key], ygt.to(device))
            loss = (loss * weight_tensor).mean()
        elif key=='sev':
            # Count number of unique label
            samples = [25702,	43652,	17571,	16998,	24570,	7579]

            ygt=data['labels'][f'label_{key}']
            weight_tensor =  torch.FloatTensor(1.0 - (samples/ np.sum(samples))).to(device)
            loss = loss_func(outputs[key], ygt.to(device))
            loss = (loss * weight_tensor).mean()


        losses += loss
    return losses

def pytorch_train_one_epoch(pytorch_network, optimizer, loss_function, sigma, scheduler):
    """
    Trains the neural network for one epoch on the train DataLoader.

    Args:
        pytorch_network (torch.nn.Module): The neural network to train.
        optimizer (torch.optim.Optimizer): The optimizer of the neural network
        loss_function: The loss function.
        sigma : sigma of the gaussian_filter1d function.
        scheduler : Scheduler.

    Returns:
        Average of the losses on the train DataLoader.
    """

    pytorch_network.train(True)

    if scheduler:
        scheduler.step()

    with torch.enable_grad():
        loss_sum = 0.
        acc_sum = 0.
        example_count = 0
        for batch, id in train_loader:
            # get the inputs - spectral sequence
            x= batch['sequence']
            # Transfer batch on GPU if needed.
            x = x.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = pytorch_network(x)
            loss = criterion_guauss_w(loss_function,y_pred, batch, sigma=sigma)
            loss.backward()
            optimizer.step()

            # Since the loss are averages for the batch, we multiply
            # it by the the number of examples
            loss_sum += float(loss) * len(x)
            example_count += len(x)

    avg_loss = loss_sum / example_count
    print('avg_loss', avg_loss)

    return avg_loss

def pytorch_test(pytorch_network, loader, loss_function, sigma):
    """
    Tests the neural network on a DataLoader.

    Args:
        pytorch_network (torch.nn.Module): The neural network to test.
        loader (torch.utils.data.DataLoader): The DataLoader to test on.
        loss_function: The loss function.
        sigma : sigma of the gaussian_filter1d function.

    Returns:
        A tuple of Average of the losses on the DataLoader, the prediction, the true label and the corresponding id.
    """

    ids=[]
    pred = []
    true = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    pytorch_network.eval()
    with torch.no_grad():
        loss_sum = 0.
        example_count = 0

        for batch, id in loader:
            # get the inputs - spectral sequence
            x= batch['sequence']

            # Transfer batch on GPU if needed.
            x = x.to(device)

            # calculate outputs by running sequence through the network
            y_pred = pytorch_network(x)

            # calculate the loss
            loss = criterion_guauss_w(loss_function,y_pred, batch, sigma=sigma)


            # Organize data prediction and true label output
            y_pred_all = []
            y_all = []
            for i, key in enumerate(y_pred):
                predkey=y_pred[key].cpu()
                predkey = nn.Sigmoid()(predkey) #value is reduced between 0 and 1 as inside de loss function BCEWithLogitsLoss
                target=batch['labels'][f'label_{key}'].cpu()
                y_pred_all.append(predkey)
                y_all.append(target)

            y_pred_all_ = np.hstack(y_pred_all)
            y_all_ = np.hstack(y_all)

            pix_id = id
            pix_id = np.reshape(pix_id,(len(y_all_),1)).cpu().numpy()

            for i in range(len(y_all_)):
                 pred.append(y_pred_all_[i])
                 true.append(y_all_[i])
                 ids.append(pix_id[i])


            # Since the loss are averages for the batch, we multiply
            # it by the the number of examples
            loss_sum += float(loss) * len(x)
            example_count += len(x)


    avg_loss = loss_sum / example_count

    return avg_loss ,pred, true,  ids



def pytorch_train(pytorch_network):
    """
    This function transfers the neural network to the right device,
    trains it for a certain number of epochs, tests at each epoch on
    the validation set and outputs the results on the test set at the
    end of training.

    Args:
        pytorch_network (torch.nn.Module): The neural network to train.


    """
    print(pytorch_network)

    # Transfer weights on GPU if needed.
    pytorch_network.to(device)

    # Define a Loss function, optimizer and scheduler
    optimizer = torch.optim.Adam(pytorch_network.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    loss_function= nn.BCEWithLogitsLoss(reduction='none').to(device)


    l_train_loss = []
    l_valid_loss = []

    # loop over the dataset multiple times
    for epoch in range(1, num_epochs + 1):
        # Print Learning Rate
        sigma=(num_epochs/2)*math.exp(-epoch/10)

        print('sigma', sigma)
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        # Training the neural network via backpropagation
        train_loss= pytorch_train_one_epoch(pytorch_network, optimizer, loss_function, sigma, scheduler)

        # Validation at the end of the epoch
        valid_loss, pred, true,  ids= pytorch_test(pytorch_network, valid_loader, loss_function, sigma)

        print("Epoch {}/{}: loss: {}, val_loss: {}, ".format(
            epoch, num_epochs, train_loss, valid_loss,         ))

        scheduler.step()
        l_train_loss.append(train_loss)
        l_valid_loss.append(valid_loss)


    # Test at the end of the training
    test_loss,  pred, true,  ids = pytorch_test(pytorch_network, test_loader, loss_function, sigma)
    print('Test:\n\tLoss: {} \n\t'.format(test_loss))


    # Save result
    id_result = np.hstack((np.stack(true, axis=0), np.stack(pred, axis=0), ids))
    pd.DataFrame(id_result).to_csv("/results/"+name+"id_result.csv")

    fig, axes = plt.subplots(2, 1)
    plt.tight_layout()

    axes[1].set_title('Train loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].plot(l_train_loss, label='Train')
    axes[1].plot(l_valid_loss, label='Validation')

    plt.show()
    plt.savefig("/results/"+name+"_graph.png")



if __name__ == '__main__':
    np.random.seed(9)
    random.seed(15)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()


    # Get label file : Multi label for each pixel Id
    labelfile = '/Alltypo_class_multi.csv'

    # Get sequence files : CSV line where each line is a pixel, and raw are spectral having the following format: [ID, 1985_B1, 1985_B2, 1985_B3....., 2010_B4, 2021_B5, 2021_B6]

    pathValid = '/Landsat_CFL_CSV/valid/*.csv' #  path with dataset using for validation
    validfile = glob(pathValid,  recursive=True)
    #validfile = pd.read_csv(validfile_)
    print(len(validfile))

    datafolder = '/Landsat_CFL_CSV/k/*.csv' # path with dataset using for training and testing
    datafile = glob(datafolder,  recursive=True)
    random.shuffle(datafile)
    print(len(datafile))

    #  Selection of the fold using cross validation
    n=5
    for i in range(0, len(datafile)-n+1, n):

        foldnum=str(int(i/5))
        print('num k fold',foldnum)
        pathtest = datafile[i:i+n]
        pathTrain = list(set(datafile) - set(pathtest))

        testfile=[]
        for f in pathtest:
            a = glob(f,  recursive=True)
            testfile.extend(a)
            # print('testfile',testfile)

        trainfile=[]
        for f in pathTrain:
            a = glob(f,  recursive=True)
            trainfile.extend(a)

        # Calling the dataset function
        train_dataset = myDataset(trainfile, labelfile)
        print("train_dataset", len(train_dataset))
        test_dataset = myDataset(testfile, labelfile)
        print("test_dataset", len(test_dataset))
        valid_dataset = myDataset(validfile, labelfile)
        print("valid_dataset", len(valid_dataset))


        # Calling the DataLoader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,  shuffle=True, num_workers=8,drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        print("test_loader", len(test_loader))

        name = 'k_'+foldnum+'_multiclass_multilabel_Trans_maxpos15_CFLimg_nodata_w_epc'

        # Choice of model to use : TempCNN or Transformer
        #model = TempCNN(input_dim=len(FEATURE_COLUMS), n_type=7, n_sev=6, n_date=38, sequencelength=37)
        model = TransformerModel(input_dim=len(FEATURE_COLUMS), n_type=7, n_sev=6, n_date=38, window_size=(1,38))


        # Train
        pytorch_train(model)
        end.record()
        torch.cuda.synchronize()

        print('Time, minutes', start.elapsed_time(end))  # milliseconds

        # Save the model
        PATH = "/results/"+name+".pt"
        torch.save(model, PATH)

