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

from models.TempCNN_subsequence import TempCNN
from models.TransformerModel_relative_subsequence import TransformerModel




print(torch.cuda.is_available())

cuda_device = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
learning_rate = 0.001
num_epochs = 50
batch_size = 32


def pytorch_accuracy(y_pred, y_true):
    """
    Computes the accuracy for a batch of predictions

    Args:
        y_pred (torch.Tensor): the logit predictions of the neural network.
        y_true (torch.Tensor): the ground truths.

    Returns:
        The average accuracy of the batch.
    """
    y_pred = y_pred.argmax(1)
    return (y_pred == y_true).float().mean() * 100
    
    
def pytorch_train_one_epoch(pytorch_network, optimizer, loss_function, scheduler):
    """
    Trains the neural network for one epoch on the train DataLoader.

    Args:
        pytorch_network (torch.nn.Module): The neural network to train.
        optimizer (torch.optim.Optimizer): The optimizer of the neural network
        loss_function: The loss function.
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
            # Transfer batch on GPU 
            x = x.to(device)
            
            ytype= batch['labels']['label_type'].to(device)
            ydate= batch['labels']['label_date'].to(device)

            # forward + backward + optimize
            y_pred = pytorch_network(x)
	    loss1= loss_function(y_pred['type'], ytype)
            loss2= loss_function(y_pred['date'], ydate)
            loss=loss1+loss2
            # zero the parameter gradients
            optimizer.zero_grad()            
            
            loss.backward()
            optimizer.step()

            # Since the loss are averages for the batch, we multiply
            # it by the the number of examples
            loss_sum += float(loss) * len(x)
            example_count += len(x)

    avg_loss = loss_sum / example_count


    return avg_loss

def pytorch_test(pytorch_network, loader, loss_function):
    """
    Tests the neural network on a DataLoader.

    Args:
        pytorch_network (torch.nn.Module): The neural network to test.
        loader (torch.utils.data.DataLoader): The DataLoader to test on.
        loss_function: The loss function.
        

    Returns:
        A tuple of Average of the losses on the DataLoader, the prediction, the true label and the corresponding id.
    """

    ids=[]
    predstype= []
    predsdate = []
    truetype = []
    truedate = []
    
    # since we're not training, we don't need to calculate the gradients for our outputs
    pytorch_network.eval()
    with torch.no_grad():
        loss_sum = 0.
        acc_sum = 0.
        example_count = 0

        for batch, id in loader:
            # get the inputs - spectral sequence
            x= batch['sequence']

            ytype=batch['labels']['label_type'].to(device)
            ydate=batch['labels']['label_date'].to(device)

            # Transfer batch on GPU if needed.
            x = x.to(device)

            # calculate outputs by running sequence through the network
            y_pred = pytorch_network(x)
            y_pred_type = pytorch_network(x)['type']
	    y_pred_date = pytorch_network(x)['date']
	    
	    predtype = torch.argmax(y_pred_type, 1)
            preddate = torch.argmax(y_pred_date, 1)
	    
	    
	    preddate = preddate.view(-1).cpu().numpy()
            preddate = np.reshape(preddate,(len(preddate),1))
           
            predtype = predtype.view(-1).cpu().numpy()
            predtype = np.reshape(predtype,(len(predtype),1))            
            
            targettype = ytype.view(-1).cpu().numpy()
            targettype = np.reshape(targettype,(len(predtype),1))

            targetdate = ydate.view(-1).cpu().numpy()
            targetdate = np.reshape(targetdate,(len(predtype),1))

            # get the loss
            loss1= loss_function(y_pred['type'], ytype)
            loss2= loss_function(y_pred['date'], ydate)
            loss=loss1+loss2	    

            # Organize data prediction and true label output
            pix_id = id
            pix_id = np.reshape(pix_id,(len(predtype),1)).cpu().numpy()

            for i in range(len(predtype)):
                 predstype.append(predtype[i])
                 predsdate.append(preddate[i])
                 truetype.append(targettype[i])
                 truedate.append(targetdate[i])
                 ids.append(pix_id[i])

            # Since the loss are averages for the batch, we multiply
            # it by the the number of examples
	    loss_sum += float(loss) * len(x)
            acc_sum += float(pytorch_accuracy(y_pred['type'], ytype)) * len(x)
            example_count += len(x)


    avg_loss = loss_sum / example_count
    avg_acc = acc_sum / example_count

    return avg_loss ,predstype, predsdate, truetype, truedate, avg_acc, ids



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
    loss_function= torch.nn.CrossEntropyLoss().to(device)


    l_train_loss = []
    l_valid_loss = []

    # loop over the dataset multiple times
    for epoch in range(1, num_epochs + 1):
        # Print Learning Rate
       
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        # Training the neural network via backpropagation
        train_loss= pytorch_train_one_epoch(pytorch_network, optimizer, loss_function,  scheduler)

        # Validation at the end of the epoch
        valid_loss, pred, true,  ids= pytorch_test(pytorch_network, valid_loader, loss_function)

        print("Epoch {}/{}: loss: {}, val_loss: {}, ".format(
            epoch, num_epochs, train_loss, valid_loss,         ))

        scheduler.step()
        l_train_loss.append(train_loss)
        l_valid_loss.append(valid_loss)


    # Test at the end of the training
    test_loss,  predstype, predsdate, truetype, truedate, test_acc, ids = pytorch_test(pytorch_network, test_loader, loss_function)
    print('Test:\n\tLoss: {} \n\t'.format(test_loss))


    # Save result
    id_result = np.hstack((np.stack(predstype, axis=0), np.stack(predsdate, axis=0), np.stack(truetype, axis=0), np.stack(truedate, axis=0)))

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



    # Get sequence files : parquet line where each line is a pixel, and raw are spectral having the following format: [ID, 1985_B1, 1985_B2, 1985_B3....., 2010_B4, 2021_B5, 2021_B6]

    pathValid = '/Landsat_CFL_CSV/valid/*.parquet' #  path with dataset using for validation
    validfile = glob(pathValid,  recursive=True)

    print(len(validfile))

    datafolder = '/Landsat_CFL_CSV/k/*.parquet' # path with dataset using for training and testing
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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,  shuffle=True, drop_last=True, num_workers=8,drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


        name = 'k_'+foldnum+'_multiclass_multilabel_Trans_maxpos15_CFLimg_nodata_w_epc'

        # Choice of model to use : TempCNN or Transformer
        #model = TempCNN(input_dim=len(FEATURE_COLUMS), n_type=11,n_date=11, sequencelength=10)
        model = TransformerModel(input_dim=len(FEATURE_COLUMS), n_type=11,  n_date=11, window_size=(1,10))


        # Train
        pytorch_train(model)
        end.record()
        torch.cuda.synchronize()

        print('Time, minutes', start.elapsed_time(end))  # milliseconds

        # Save the model
        PATH = "/results/"+name+".pt"
        torch.save(model, PATH)

