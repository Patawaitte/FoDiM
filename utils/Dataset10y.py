import torch
from torch.utils.data import Dataset
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np





def load_parquet(seq):
    data = pd.read_parquet(seq)

    data.drop(columns=['id_grid', 'index',  'randomID','AN_ORIGINE', 'ORIGINE','id_left', 'ared', 'use', 'RandomV','index_righ', 'NBR_1', 'NBR_2','NBR_3','NBR_4','NBR_5','NBR_6','NBR_7', 'NBR_8', 'NBR_9', 'NBR_10','iNBR50','iNBR100','iNBR125','iNBR150','iNBR200','geometry', 'randomAN', 'slope', 'theilslope'], inplace=True)
    data['Xseq'] = data['Xseq']+1
    data['Xseq'] = data['Xseq'].replace(100, 0)
    data=data.replace(-32768.0000, np.nan)
    return data


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
class Dataset(Dataset):

         def __init__(self, seqfile) :
             self.seqfile = load_parquet(seqfile)

             self.disId_data =  self.seqfile.iloc[:,1]  # Get the Id of the pixel
             self.disId = np.asarray(self.disId_data.values).astype(np.float)

             self.X_data = self.seqfile.iloc[:,2:62]/10000 #31=2+6band*10y
             self.X = self.X_data.values
             self.X = np.asarray(self.X, dtype='float32')
             self.seq = self.X.reshape(self.X.shape[0],int(self.X.shape[1]/6),6)  #reshape sequence as numpy of size (37,6)

             # Get the multilabel for each target
             self.Y_type = self.seqfile.iloc[:,229:229+7]  #For type (7 classes)
             self.Y_sev = self.seqfile.iloc[:,229+7:229+7+6]  #For severity (6 classes)
             self.Y_date = self.seqfile.iloc[:,229+7+6:229+7+6+1+38]  #For date (38 classes)

             self.Y_type = self.seqfile.iloc[:,0]  #For type
             self.Y_date = self.seqfile.iloc[:, -1]  #For date

             self.Y_type= np.asarray(self.Y_type.values, dtype='float32')
             self.Y_date= np.asarray(self.Y_date.values, dtype='float32')


         def __len__(self):
             return len(self.seq)

         def __getitem__(self, idx):
             labels_type=self.Y_type[idx]
             labels_date=self.Y_date[idx]

             sequence = self.seq[idx]
             sequence= numpy_fill(sequence)  #fill nan value

             # Organize the dataset into dictionnary
             data = {'sequence':torch.Tensor(sequence),
                    'labels': {'label_type':torch.tensor(labels_type).long(),
                               'label_date':torch.tensor(labels_date).long()}}

             id = self.disId[idx]
             return data, id

# Dataset Class for inference (without label)
# Input are dataframe created from slidding windows images with x,y coordinate and spectral temporal value
class Dataset_inf(Dataset):
         def __init__(self, seqdf) :
             self.seqfile = seqdf

             self.coordx =  self.seqfile.iloc[:,0] #Get x coordinate
             self.coordx = np.asarray(self.coordx).astype(np.float)

             self.coordy =  self.seqfile.iloc[:,1] #Get y coordinate
             self.coordy = np.asarray(self.coordy).astype(np.float)


             self.X_data = self.seqfile.iloc[:,2:44]/10000  #31=1+6band*7y
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
    seqfile = '/FoDiM/Dataset/k/10.0.parquet'
    validfile = glob(seqfile,  recursive=True)
    tt = Dataset(validfile)
    fig = plt.figure(figsize=(15,4))
    i=0
    for seq in tt:
        i +=1
        data, id = seq
        ax = fig.add_subplot(2,4,i)
        ax.plot((data['sequence']))
