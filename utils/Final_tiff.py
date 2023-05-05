
import numpy as np
from rasterio.crs import CRS

my_wkt="""PROJCRS["NAD_1983_Canada_Lambert",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983", ELLIPSOID["GRS 1980",6378137,298.257222101004, LENGTHUNIT["metre",1]],ID["EPSG",6269]],PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],CONVERSION["unnamed", METHOD["Lambert Conic Conformal (2SP)", ID["EPSG",9802]],PARAMETER["Latitude of false origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]], PARAMETER["Longitude of false origin",-95,ANGLEUNIT["Degree",0.0174532925199433], ID["EPSG",8822]], PARAMETER["Latitude of 1st standard parallel",49, ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",77,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1], ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1], LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""
dst_projection = CRS.from_wkt(my_wkt)


def makemap(idresult, name, output):
        """
    This function transform probabily outcome from multiclass model network to multi disturbance map using several conditions and thresholds.

    Args:
        idresult : Dataframe with x, y and result probability
        name : The name of the output.
        output: path of the output.

    Returns:
        4 maps .tif of forest disturbance
    """
    ###############Select maximum probabilty
    ################################################################################
    idresult['cdate_max'] = idresult.iloc[:,2+7+6:2+7+6+38].idxmax(1).str[4:]
    idresult['ctype_max'] = idresult.iloc[:,2:2+7].idxmax(1).str[5:]
    idresult['csev_max'] = idresult.iloc[:,2+7:2+7+6].idxmax(1).str[5:]


    ###############Select 2nd higher probability for the date
    ################################################################################
    idresult['cdate_2nd'] = (idresult.iloc[:,2+7+6:1+7+6+38].where(idresult.iloc[:,2+7+6:1+7+6+38].gt(0.1))         #  threshold to select only probability higher of 0.1 for second date
                     .stack().groupby(level=0)
                     .apply(lambda x: x.nlargest(3).index
                                       .get_level_values(1).to_list()
                           )
                  ).str[1].str[4:]
    idresult['cdate_2nd'] = idresult['cdate_2nd'].fillna(-999)

    print(idresult)
    # Condition to avoid subsequent year of disturbance detection
    idresult['cdate_2nd'] =np.where((idresult['cdate_2nd'].astype(int)==idresult['cdate_max'].astype(int)+1)| (idresult['cdate_2nd'].astype(int)==idresult['cdate_max'].astype(int)-1), -999, idresult['cdate_2nd'])

    # # Chercher vrai date si Max date =0 mais type et sev =prediction
    idresult['cdate_max'] =np.where((idresult['cdate_max'].astype(int)==0) & ((idresult['csev_max']!='WATER')& (idresult['csev_max']!='NO CHANGE')& (idresult['csev_max']!='RECOVERY')), idresult['cdate_2nd'], idresult['cdate_max'])

    # # Condition pour arreter recherche si Max date =0
    idresult['cdate_2nd'] =np.where((idresult['cdate_max'].astype(int)==0) , -999, idresult['cdate_2nd'])


    ###############CONDITIONS TO Find 2nd severity and type
    ################################################################################
    idresult['cdate_2nd'] = idresult['cdate_2nd'].replace({-999: np.nan})
    idresult['cdate_max'] = idresult['cdate_max'].replace({-999: 0})
    idresult["ctype_2nd"] = -999
    idresult["csev_2nd"] = -999


    idresult['ctype_2nd'] = (idresult.iloc[:,2:2+7].where(idresult.iloc[:,2:2+7].gt(0.6))         #  threshold to select only probability higher of 0.6 for second type
                     .stack().groupby(level=0)
                     .apply(lambda x: x.nlargest(2).index
                                       .get_level_values(1).to_list()
                           )
                  ).str[1]

    if idresult['ctype_2nd'].isnull().values.all():
        pass
    else :
        idresult['ctype_2nd'] = np.where(idresult['ctype_2nd']!=-999 , idresult['ctype_2nd'].str[5:].astype(float), -999)



    idresult['csev_2nd'] = (idresult.iloc[:,2+7:2+7+6].where(idresult.iloc[:,2+7:2+7+6].gt(0.6))         #  threshold to select only probability higher of 0.6 for second severity
                     .stack().groupby(level=0)
                     .apply(lambda x: x.nlargest(3).index
                                       .get_level_values(1).to_list()
                           )
                  ).str[1]
    if idresult['csev_2nd'].isnull().values.all():
        pass
    else :
        idresult['csev_2nd'] = np.where(idresult['csev_2nd']!=-999 , idresult['csev_2nd'].str[5:].astype(float), -999)



    # Enlever seconde date, si pas de second type
    idresult['cdate_2nd'] = np.where(idresult['cdate_2nd'].notnull() & (idresult['cdate_max'].astype(int)!=0) & idresult['ctype_2nd'].isnull(), np.nan, idresult['cdate_2nd'])

    # Recupérer type et sev si date2 n'est pas null, mais différent de 0... Les 2 perturbations sont du même type
    idresult['ctype_2nd'] = np.where(idresult['cdate_2nd'].notnull() & (idresult['cdate_max'].astype(int)!=0) & idresult['ctype_2nd'].isnull(), idresult['ctype_max'], idresult['ctype_2nd'])
    idresult['csev_2nd'] = np.where(idresult['cdate_2nd'].notnull() & (idresult['cdate_max'].astype(int)!=0) & idresult['csev_2nd'].isnull(), idresult['csev_max'], idresult['csev_2nd'])


    ###############CONDITIONS EPIDEMIC == PROGRESSIVE
    ################################################################################

    idresult['csev_max'] =np.where((idresult['ctype_max']=='1') , '2', idresult['csev_max'])
    idresult['csev_2nd'] =np.where((idresult['ctype_2nd']=='1') , '2', idresult['csev_2nd'])
    #

    ###############TABLE MANIPULATION TO ORDER EVENT BY YEAR
    ################################################################################

    frametocompare = idresult[['x', 'y','cdate_max', 'ctype_max', 'csev_max', 'cdate_2nd', 'ctype_2nd', 'csev_2nd']]

    frametocompare['dis_date_1'] = np.where(frametocompare['cdate_2nd']<frametocompare['cdate_max'], frametocompare['cdate_2nd'], frametocompare['cdate_max'])
    frametocompare['dis_type_1'] = np.where(frametocompare['cdate_2nd']<frametocompare['cdate_max'], frametocompare['ctype_2nd'], frametocompare['ctype_max'])
    frametocompare['dis_sev_1'] = np.where(frametocompare['cdate_2nd']<frametocompare['cdate_max'], frametocompare['csev_2nd'], frametocompare['csev_max'])

    #
    frametocompare['dis_date_2'] = np.where(frametocompare['cdate_2nd']<frametocompare['cdate_max'], frametocompare['cdate_max'], frametocompare['cdate_2nd'])
    frametocompare['dis_type_2'] = np.where(frametocompare['cdate_2nd']<frametocompare['cdate_max'], frametocompare['ctype_max'], frametocompare['ctype_2nd'])
    frametocompare['dis_sev_2'] = np.where(frametocompare['cdate_2nd']<frametocompare['cdate_max'], frametocompare['csev_max'], frametocompare['csev_2nd'])



    ############### MAKE NEW RASTER IMAGES
    ################################################################################

    bands=['dis_date_1', 'dis_type_1','dis_sev_1','dis_date_2', 'dis_type_2', 'dis_sev_2']
    for b in bands:

        new_df = frametocompare[['y','x', b]].copy()
        new_df[b]=new_df[b].astype(float)
        print(new_df.shape)

        da = new_df.set_index(['y', 'x']).to_xarray()
        da = da.set_coords(['y', 'x'])
        da = da.rio.write_crs(dst_projection, inplace=True)
        da.rio.to_raster(output+'result_'+name+'_'+b+'.tif')




