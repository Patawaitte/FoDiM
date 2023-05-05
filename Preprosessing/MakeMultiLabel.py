import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()


df = gpd.read_file('/point.shp')

type_per = pd.read_csv('/typology.csv', sep=',')
frame = df.merge(type_per[['old_id', 'id', 'Severity_id', 'Type_id']],left_on=('dis1_id'),right_on=('old_id'),how='left').rename(columns = {'id': 'id_dis1', 'Severity_id': 'Sevid_dis1', 'Type_id': 'Typeid_dis1'})
frame2 = frame.merge(type_per[['old_id', 'id', 'Severity_id', 'Type_id']],left_on=('dis2_id'),right_on=('old_id'),how='left').rename(columns = {'id': 'id_dis2', 'Severity_id': 'Sevid_dis2', 'Type_id': 'Typeid_dis2'})

frame_sev= frame2[['pix_id','Sevid_dis1', 'Sevid_dis2']]
frame_typ= frame2[['pix_id','Typeid_dis1', 'Typeid_dis2']]
frame_date= frame2[['pix_id','dis1_y', 'dis2_y']]

frame_sev['combined'] = [[e for e in row if e==e] for row in frame_sev[['Sevid_dis1', 'Sevid_dis2']].values.tolist()]
frame_typ['combined'] = [[e for e in row if e==e] for row in frame_typ[['Typeid_dis1', 'Typeid_dis2']].values.tolist()]
frame_date['combined'] = [[e for e in row if e==e] for row in frame_date[['dis1_y', 'dis2_y']].values.tolist()]


hotsev = mlb.fit(frame_sev['combined'])
hotsev = mlb.transform(frame_sev['combined'])

hotyp = mlb.fit(frame_typ['combined'])
hotyp = mlb.transform(frame_typ['combined'])

hotdate = mlb.fit(frame_date['combined'])
hotdate = mlb.transform(frame_date['combined'])

list_sev = ['sev{}'.format(id_idx) for id_idx in range(0,6)]
list_type = ['type{}'.format(id_idx) for id_idx in range(0,8)]
list_type.remove('type2')

date_list_all = ['y_{}'.format(id_idx) for id_idx in range(1985,2022)]
date_list_all_= date_list_all.insert(0,'no_dis')


dfsev = pd.DataFrame(hotsev, columns = list_sev)
finalsev= pd.concat( [frame_sev[['pix_id']], dfsev], axis=1 )

dftype = pd.DataFrame(hotyp, columns = list_type)
finaltype= pd.concat( [frame_typ[['pix_id']], dftype], axis=1 )

dfdate = pd.DataFrame(hotdate, columns = date_list_all_)
finaldate= pd.concat( [frame_date[['pix_id']], dfdate], axis=1 )


alllabel = pd.merge(finaltype, finalsev, how='inner', left_on = 'pix_id',  right_on = 'pix_id')
alllabel2 = pd.merge(alllabel, finaldate, how='inner', left_on = 'pix_id',  right_on = 'pix_id')

alllabel2.to_csv('/Alltypo_class_multi.csv')


finalsev.to_csv('/Severity_class_multi.csv')
finaltype.to_csv('/Type_class_multi.csv')
finaldate.to_csv('/Date_class_multi.csv')



