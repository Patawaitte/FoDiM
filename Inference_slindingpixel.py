import os
import rioxarray
import time
from Inference.Final_tiff import makemap
import torch.nn as nn

from utils.dataset_multi_label_class import MultiDisDataset_inf as myDataset
from torch.utils.data.dataset import Subset
from glob import glob
import torch
import numpy as np
import pandas as pd

import rasterio
from tqdm import tqdm
import warnings
import cProfile
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


start_time = time.time()

# Dataset to use
IMG_PATH="/LANDSAT_CLIP/*.tif"  # Landsat time serie image for disturbance segmentation
PATH = "/mymodel.pt" # Model
outputpath = '/results/Resultmap.tif/'
if not os.path.exists(outputpath):
    os.mkdir(outputpath)

# Load model to do inference
model = torch.load(PATH)
model.eval()

cuda_device = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(pytorch_network, loader):
    """
    This function transfers the neural network to the right device,
    apply the network to the data and get the probability result.

    Args:
        pytorch_network (torch.nn.Module): The neural network to apply.
        loader (torch.utils.data.DataLoader): The DataLoader to infer on.

    Returns:
        Array with probability for all classes and targets with x,y coordinate.

    """
    cx=[]
    cy=[]

    preds=[]
    pytorch_network.eval()
    with torch.no_grad():
        for batch, pix_x, pix_y in loader:
            x= batch['sequence']

            # Transfer batch on GPU if needed.
            x = x.to(device)

            # Get the prediction of the network
            y_pred = pytorch_network(x)

            # Organize the output
            y_pred_all = []
            for i, key in enumerate(y_pred):
                predkey=y_pred[key].cpu()
                predkey = nn.Sigmoid()(predkey) #value is reduced between 0 and 1 as inside de loss function BCEWithLogitsLoss
                y_pred_all.append(predkey)
            y_pred_all_ = np.hstack(y_pred_all)
            y_pred_all_s= np.squeeze(y_pred_all_)

            preds.append(y_pred_all_s)
            cx.append(pix_x.item())
            cy.append(pix_y.item())


        final_result = np.column_stack((cx, cy, preds))


    return final_result

###############################################################
    ####### Read image with slidding windows


dffinal = pd.DataFrame()
dffinalall = []

lt_bands = glob(IMG_PATH)
lt_bands.sort() # sorted list
template= rasterio.open(lt_bands[0]) # Image used to get the shape

w, h = template.shape # Get the shape

## prepare different n_wins window offsets to be read
# inspired by : https://www.kaggle.com/code/quandapro/sliding-window-with-importance-map-inference
w_h=1000
w_w=1000
s_h=1000 #for overlapping
s_w=1000 #for overlapping

# Generate a list of starting points a.k.a top left of window
starting_points = [(x, y)  for x in sorted(set( list(range(0, h - w_h, s_h)) + [h - w_h] ))
                                   for y in sorted(set( list(range(0, w - w_w, s_w)) + [w - w_w] ))]

# Loop through all starting points
for i in tqdm(range(len(starting_points))):
    x, y = starting_points[i]
    nameexport= (str(x) +'_'+ str(y))

    for band in tqdm(lt_bands):
        nband = band[-6:-4]  #name of the band
        nyear = band[-19:-15] #year
        cname=nband+'_'+nyear
        rds = rioxarray.open_rasterio(band).isel(x=slice(x, x+w_h), y=slice(y, y+w_w), band=0) #read the image

        rds = rds.squeeze().drop("spatial_ref").drop("band")
        rds.name = "data"
        res=rds.to_dataframe().reset_index()


        dffinal['x']=res['x']  # get x
        dffinal['y']=res['y'] # get y
        dffinal[cname]=res['data'] # get the spectral sequence



    test_dataset = myDataset(dffinal) #Call dataset
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size) #Call dataloader
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()


    result = inference(model, test_loader) # Inference with the network
    print(result)
    end.record()
    torch.cuda.synchronize()

###############################################################
    ####### Transform the probability result to disturbances maps

    # Add column names
    labels_sev = ['0', '1', '2', '3', '4', '5']
    labels_type  =['0', '1' ,'3', '4','5','6', '7']
    labels_date  =[ 0,   1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996,
    1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    allcolumnlist = ['x']+['y']+['predt' + sub for sub in labels_type]+['preds' + sub for sub in labels_sev]+['pred' + sub for sub in map(str, labels_date)]
    allresultdf=pd.DataFrame(result,columns =allcolumnlist)

    # Get the mean result of duplicate pixel  s
    allresultdf_ = allresultdf.groupby(['x', 'y'], as_index=False).mean()

    # Export the probability result (for statistique analyse)
    pd.DataFrame(allresultdf_).to_csv(outputpath+nameexport+'.csv', index=False)

    # Apply makemap function to transform df to .tif map
    makemap(allresultdf_, nameexport, outputpath)

    #









print("Process finished --- %s seconds ---" % (time.time() - start_time))



