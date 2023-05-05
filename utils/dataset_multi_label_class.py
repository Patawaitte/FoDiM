import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np




def load_csv(seqcsv, labelfile):
    """
    Load the csv to make sequence and label ready to dataset
    Args:
        seqcsv : csv file. Where each line is a pixel, and raw are spectral having the following format: [ID, 1985_B1, 1985_B2, 1985_B3....., 2010_B4, 2021_B5, 2021_B6]
        labelfile : TMulti label for each pixel Id.

    Returns:
       Dataframe with spectral information and label concatenated
    """
    list = []
    for csvfile in seqcsv :
        data = pd.read_csv(csvfile)
        label = pd.read_csv(labelfile)
        data.drop_duplicates()
        datalabl = pd.merge(data, label, how='inner', left_on = 'id',  right_on = 'pix_id')
        list.append(datalabl)
    idresult = pd.concat(list, axis=0, ignore_index=True)

    return idresult



def numpy_fill(a, startfillval=0):
    """
    Fill sequence with the previous year value when the pixel are null.
    https://stackoverflow.com/questions/62038693/numpy-fill-nan-with-values-from-previous-row
    Args:
        a : array corresponding to the sequence
        startfillval : 0

    Returns:
       Array without nan value
    """
    mask = np.isnan(a)
    tmp = a[0].copy()
    a[0][mask[0]] = startfillval
    mask[0] = False
    idx = np.where(~mask,np.arange(mask.shape[0])[:,None],0)
    out = np.take_along_axis(a,np.maximum.accumulate(idx,axis=0),axis=0)
    a[0] = tmp
    return out

# Dataset Class for training and test
class MultiDisDataset(Dataset):

         def __init__(self, seqfile, labelfile) :
             self.seqfile = load_csv(seqfile, labelfile)

             self.disId_data =  self.seqfile.iloc[:,3]  # Get the Id of the pixel
             self.disId = np.asarray(self.disId_data.values).astype(np.float)

             self.X_data = self.seqfile.iloc[:,5:227]/10000  #227=5+6bands*37y get the spectral value sequences ..../1000 is for normalized the spectral value
             self.X = self.X_data.values
             self.X = np.asarray(self.X, dtype='float32')
             self.seq = self.X.reshape(self.X.shape[0],int(self.X.shape[1]/6),6)  #reshape sequence as numpy of size (37,6)

             # Get the multilabel for each target
             self.Y_type = self.seqfile.iloc[:,229:229+7]  #For type (7 classes)
             self.Y_sev = self.seqfile.iloc[:,229+7:229+7+6]  #For severity (6 classes)
             self.Y_date = self.seqfile.iloc[:,229+7+6:229+7+6+1+38]  #For date (38 classes)

             self.Y_type= np.asarray(self.Y_type.values, dtype='float342')
             self.Y_sev= np.asarray(self.Y_sev.values, dtype='float32')
             self.Y_date= np.asarray(self.Y_date.values, dtype='float32')


         def __len__(self):
             return len(self.seq)

         def __getitem__(self, idx):
             labels_type=self.Y_type[idx]
             labels_sev=self.Y_sev[idx]
             labels_date=self.Y_date[idx]

             sequence = self.seq[idx]
             sequence[sequence == -3.2768] = np.nan
             sequence= numpy_fill(sequence)  #fill nan value

             # Organize the dataset into dictionnary
             data = {'sequence':torch.Tensor(sequence),
                    'labels': {'label_type':torch.tensor(labels_type).float(),
                               'label_sev':torch.tensor(labels_sev).float(),
                               'label_date':torch.tensor(labels_date).float()}}

             id = self.disId[idx]
             return data, id

# Dataset Class for inference (without label)
# Input are dataframe created from slidding windows images with x,y coordinate and spectral temporal value
class MultiDisDataset_inf(Dataset):
         def __init__(self, seqdf) :
             self.seqfile = seqdf

             self.coordx =  self.seqfile.iloc[:,0] #Get x coordinate
             self.coordx = np.asarray(self.coordx).astype(np.float)

             self.coordy =  self.seqfile.iloc[:,1] #Get y coordinate
             self.coordy = np.asarray(self.coordy).astype(np.float)


             self.X_data = self.seqfile.iloc[:,2:224]/10000  #224=2+6*37 ..... /1000 is for normalized the spectral value
             self.X = self.X_data.values
             self.X = np.asarray(self.X, dtype='float32')
             self.seq = self.X.reshape(self.X.shape[0],int(self.X.shape[1]/6),6) #reshape sequence as numpy of size (37,6)



         def __len__(self):
             return len(self.seq)
         def __getitem__(self, idx):

             sequence = self.seq[idx]
             sequence[sequence == -3.2768] = np.nan
             sequence= numpy_fill(sequence) #fill nan value

             # Organize the dataset into dictionnary
             data = {'sequence':torch.Tensor(sequence),
                     }

             cX = self.coordx[idx]
             cY = self.coordy[idx]

             return data, cX, cY


if __name__=="__main__":
    # Test for visualization
    seqfile = '/test/*.csv'
    validfile = glob(seqfile,  recursive=True)
    labels= '/Alltypo_class_multi.csv'
    tt = MultiDisDataset(validfile,labels)
    fig = plt.figure(figsize=(15,4))
    i=0

    for seq in tt:
        i +=1
        data, id = seq
        ax = fig.add_subplot(2,4,i)
        ax.plot((data['sequence']))
