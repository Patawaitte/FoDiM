
import numpy as np
from rasterio.crs import CRS

my_wkt="""PROJCRS["NAD_1983_Canada_Lambert",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983", ELLIPSOID["GRS 1980",6378137,298.257222101004, LENGTHUNIT["metre",1]],ID["EPSG",6269]],PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],CONVERSION["unnamed", METHOD["Lambert Conic Conformal (2SP)", ID["EPSG",9802]],PARAMETER["Latitude of false origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]], PARAMETER["Longitude of false origin",-95,ANGLEUNIT["Degree",0.0174532925199433], ID["EPSG",8822]], PARAMETER["Latitude of 1st standard parallel",49, ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",77,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1], ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1], LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""
dst_projection = CRS.from_wkt(my_wkt)


def makemap(idresult, name, output):
  """
    This function transform pixel outcome to raster imagery.

    Args:
        idresult : Dataframe with x, y and classification result
        name : The name of the output.
        output: path of the output.

    Returns:
        Two maps .tif of forest disturbance type and year
            """
    frametocompare = idresult[['x', 'y','type',  'Year']]
    bands=['type', 'Year']
    for b in bands:

        new_df = frametocompare[['y','x', b]].copy()
        new_df[b]=new_df[b].astype(float)


        da = new_df.set_index(['y', 'x']).to_xarray()
        da = da.set_coords(['y', 'x'])
        da = da.rio.write_crs(dst_projection, inplace=True)
        da.rio.to_raster(output+'result_'+name+'_'+b+'.tif') # Save the map



