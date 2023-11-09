import os
import rioxarray
import time
from Inference.Final_tiff import makemap
import torch.nn as nn

from utils.dataset_subset import MultiDisDataset_inf as myDataset
from torch.utils.data.dataset import Subset
from glob import glob
import torch
import numpy as np
import pandas as pd

import rasterio
from tqdm import tqdm
import warnings
import cProfile

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

    predstype= []
    predsdate = []
    
    pytorch_network.eval()
    with torch.no_grad():
        for batch, pix_x, pix_y in loader:
            x= batch['sequence']

            # Transfer batch on GPU if needed.
            x = x.to(device)

            # Get the type prediction of the network and reshape it
            y_pred_type = pytorch_network(x)['type']
            predtype = torch.argmax(y_pred_type, 1)
            predtype = predtype.view(-1).cpu().numpy()
            predtype = np.reshape(predtype,(len(predtype),1))

	    # Get the date prediction of the network and reshape it
            y_pred_date = pytorch_network(x)['date']
            preddate = torch.argmax(y_pred_date, 1)
            preddate = preddate.view(-1).cpu().numpy()            
            preddate = np.reshape(preddate,(len(preddate),1))



            # Organize the output
            for i in range(len(predtype)):
                 predstype.append(predtype[i])
                 predsdate.append(preddate[i])

                 cx.append(pix_x[i])
                 cy.append(pix_y[i])



        final_result = np.column_stack((cx, cy, predstype, predsdate))


    return final_result

###############################################################
    ####### Read image with slidding windows



lt_bands = glob(IMG_PATH)
lt_bands.sort() # sorted list


##list date to use for each subsequence
lt_bands1 = [s for s in lt_bands if "1985" in s or "1986" in s or "1987" in s or "1988" in s or "1989" in s or "1990" in s or "1991" in s or "1992" in s or "1993" in s or "1994" in s]
lt_bands2 = [s for s in lt_bands if "1994" in s or "1995" in s or "1996" in s or "1997" in s or "1998" in s or "1999" in s or"2000" in s or "2001" in s or "2002" in s or "2003" in s]
lt_bands3 = [s for s in lt_bands if "2003" in s or "2004" in s or "2005" in s or "2006" in s or "2007" in s or "2008" in s or "2009" in s or "2010" in s or "2011" in s or "2012" in s]
lt_bands4 = [s for s in lt_bands if "2012" in s or "2013" in s or "2014" in s or "2015" in s or "2016" in s or "2017" in s or "2018" in s or "2019" in s or "2020" in s or "2021" in s]
allwindlist= [lt_bands1,lt_bands2,lt_bands3,lt_bands4]
namewindow=('1985_1994', '1994_2003', '2003_2012', '2012_2021')

windows_dict = {name: lst for name, lst in zip(namewindow, allwindlist)}
windows_dict_last = list(windows_dict.items())

## Loop over the subsequences
for name, value in windows_dict_last:
    firstyear= name.split('_')[0]

	
    dffinal = pd.DataFrame()
    dffinalall = []
    template= rasterio.open(value[0])
    w, h = template.shape

    ## prepare different n_wins window offsets to be read
    # inspired by : https://www.kaggle.com/code/quandapro/sliding-window-with-importance-map-inference
    w_h=2048
    w_w=2048
    s_h=2048
    s_w=2048

    # Generate a list of starting points a.k.a top left of window
    starting_points = [(x, y)  for x in sorted(set( list(range(0, h - w_h, s_h)) + [h - w_h] ))
                                       for y in sorted(set( list(range(0, w - w_w, s_w)) + [w - w_w] ))]
   # Loop through all starting points
    for i in tqdm(range(len(starting_points))):

        x, y = starting_points[i]
        nameexport= name+'_'+str(x) +'_'+ str(y)

	
        for band in tqdm(value):
            nband = band[-6:-4]  #name of the band
            nyear = band[-19:-15] #year
            cname=nband+'_'+nyear
            rds = rioxarray.open_rasterio(band).isel(x=slice(x, x+w_h), y=slice(y, y+w_w), band=0)
           

            rds = rds.squeeze().drop("spatial_ref").drop("band")
            rds.name = "data"

            res=rds.to_dataframe().reset_index()


            dffinal['x']=res['x']
            dffinal['y']=res['y']
            dffinal[cname]=res['data'] # get the spectral sequence
 

        test_dataset = myDataset(dffinal) #Call dataset
        batch_size = 512
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size) #Call dataloader
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()



        result = inference(model, test_loader)  # Inference with the network

      
        end.record()
        torch.cuda.synchronize()


        allcolumnlist =('x', 'y', 'type', 'rYear')
        allresultdf=pd.DataFrame(result,columns =allcolumnlist)
        allresultdf_ = allresultdf.groupby(['x', 'y'], as_index=False).mean()
        allresultdf_['type'] = allresultdf_['type'].round(0).astype(int)
        allresultdf_['Year'] = allresultdf_['rYear']+int(firstyear)-1
        allresultdf_['Year'] = np.where(allresultdf_['Year']== (int(firstyear)-1), 99 ,allresultdf_['Year']) # Transform relative year to actual year

 	# Apply makemap function to transform df to .tif map
        makemap(allresultdf_, nameexport, outputpath)
 
        #












