import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import requests
import time

values_csv=[]
cols_drop_csv=[]
def add_to_csv_values(col,val,meaning):
  values_csv.append((col,val,meaning))
  # if col in values_csv.keys():
  #   values_csv[col].append({val:meaning})
  # else:
  #   values_csv[col]=[{val:meaning}]

def add_to_col_drop(col):
  cols_drop_csv.append(col)

def number_encode_features(df,feature,mapping):
    result = df.copy() # take a copy of the dataframe
    map = []
    for x in df[feature]:
      map.append(mapping[x])
    result[feature+'_encoding'] = np.array(map)
    return result

def generateLabelsMapping(df,feature):
  x = df[feature].unique()
  map = {}
  n = 0
  for i in x:
    map[i] = n
    n+=1
  return map

def add_mapping_to_csv(feature,mapping):
  for k in mapping.keys():
    add_to_csv_values(feature+'_encoding',mapping[k],k)



def runExtract(dataSet_csv,lookupPath,city_csv_path,lookup_path):
  df= pd.read_csv(dataSet_csv,index_col=0)
  
  df_lookup = pd.read_csv(lookupPath)
  
  for i in range(len(df_lookup)):
    temp = df_lookup.iloc[i]
    values_csv.append((temp['column'],temp['value'],temp['meaning']))

  final_city =[]
  for i in range(0,len(df)):
    final_city.append("no")
  district = df.local_authority_district_encoding
  longitude = df.longitude
  latitude = df.latitude
  map ={}
  keys = ["7152c9544fa9d49dc8e326d59812cd67","fff335435883a23f55f9c40c9d24c37a","d71dd7bba01b87810f5d394def2149a6","dc2de8d3a1d4789af2eae7d87aebd0df","ede1db1243972468bf8148789a9772ab","4149cb4aa072ef26d5900996a937e9e1","c3c2ced13d845a6dec601494457daf27","c52a78ec9a4cd697a823eca34078b26f","fe749bfabcaa6e0da313054f62997b97","1c1d00dc4ab29e4225c051e10063aeed","54f32acfd5388d6320717409da0f13d1"]
  index_of_key=1
  c = 0
  tries = 0
  for i in tqdm(range(0,len(df))):
    if(district.iloc[i] in map):
      final_city[i] = map[district.iloc[i]]
    else:
      try:
        api_url = 'http://api.openweathermap.org/geo/1.0/reverse?lat='+str(latitude.iloc[i])+'&lon='+str(longitude.iloc[i])+'&limit=5&appid='+keys[index_of_key]
        response = requests.get(api_url).json()
        final_city[i] = response[0]['name']
        time.sleep(1)
        map[district.iloc[i]] = final_city[i]
        tries = 0
      except:
        index_of_key+=1
        index_of_key%=10
        print(response)
        c+=1
  print("except",c)
  df['city'] = final_city
  df = number_encode_features(df,'city',generateLabelsMapping(df,'city'))
  add_mapping_to_csv('city',generateLabelsMapping(df,'city'))
  add_to_col_drop('city')
  df = df.drop(cols_drop_csv,axis=1)
  np.savetxt(lookup_path, values_csv, delimiter=',', fmt=['"%s"' , '"%s"', '"%s"'], header='column,value,meaning', comments='')
  df.to_csv(city_csv_path,index=True)
 