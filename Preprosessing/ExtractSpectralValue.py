import geopandas as gpd
from glob import glob
import os
import numpy as np
import museotoolbox
import pandas as pd
from tqdm import  tqdm


# Path images Landsat - One images by years and by bands
pathcomposite = "/LANDSAT_COMPOSITE_CFL"

# final csv
pathcsv = "/Landsat_CFL_CSV/"
os.makedirs(pathcsv,exist_ok=True)

dem_fps = glob(os.path.join(pathcomposite, "*.tif"))
dem_fps.sort()
year=[]
poly=[]

# Read the shapefile with reference points
randompt = gpd.read_file("points.shp", index=1)
randompt['idclass']=randompt['idclass'].astype(int)


# museotoolbox works with gpkg, transforming shapefile to gpkg
randompt.to_file('/points.gpkg',driver='GPKG')
all=[]
dffinal = pd.DataFrame(columns=['id_code'])

# Loop through  Landsat images and extract spectral value for each point
for fp in tqdm(dem_fps):

    band = fp[-6:-4]
    year = fp[-19:-15]
    cname=band+'_'+year

    X,y, an_code, id_code, dis_code, typesample  = museotoolbox.processing.extract_ROI(fp, '/points.gpkg', 'pix_id','an_code','idclass','DistrubtId', 'cross_samp', prefer_memory=True)

    bands = np.array([]).reshape(-1,X.shape[1])
    reshape_1d_vertical = lambda arr : np.asarray(arr).reshape(-1,1)

    columns = ['year', 'dis_code', 'an_code','id_code','id', 'cross_samp', band]
    an_codes = []
    id_codes = []
    dis_codes = []
    sample_codes = []

    ordered_y = sorted(np.unique(y))

    for y_unique in ordered_y:
        bands_values = X[np.where(y==y_unique),:]
        mean_per_band = np.median(bands_values,axis=1) # axis = 1 , par colonne, donc par bande
        bands = np.vstack((bands,mean_per_band))
        dis_codes.append( dis_code[np.where(y==y_unique)][0])
        an_codes.append( an_code[np.where(y==y_unique)][0])
        id_codes.append( id_code[np.where(y==y_unique)][0])
        sample_codes.append( typesample[np.where(y==y_unique)][0])


    final_result = np.hstack((
        reshape_1d_vertical([int(year)]*len(ordered_y)),
        reshape_1d_vertical(dis_codes),
        reshape_1d_vertical(an_codes),
        reshape_1d_vertical(id_codes),
        reshape_1d_vertical(ordered_y),
        reshape_1d_vertical(sample_codes),
        bands))
    print(final_result[0])
    final_df = pd.DataFrame(final_result,columns=columns)
    final_df[['year', 'dis_code','an_code','id_code','id','cross_samp']] = final_df[['year', 'dis_code','an_code','id_code','id', 'cross_samp']].apply(pd.to_numeric,downcast='integer')


    dffinal['id_code']= final_df['id_code']
    dffinal['dis_code']= final_df['dis_code']
    dffinal['year']= final_df['year']
    dffinal['id']= final_df['id']
    dffinal['cross_samp']= final_df['cross_samp']
    dffinal[cname]= final_df[band]


# Export csv depending of the cross validation grid ID
for (cross), group in tqdm(dffinal.groupby(['cross_samp'])):
     cross_samp = group['cross_samp'].max()

     group.to_csv(pathcsv+f'{cross}.csv', index=False)



