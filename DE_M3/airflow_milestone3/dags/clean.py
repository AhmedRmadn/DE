from datetime import date, datetime
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from scipy import stats
from tqdm import tqdm





    


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


def trunk_road_flag_missing(df,missing='Data missing or out of range',default = 'Non-trunk'):
  res = []
  values = df.trunk_road_flag.unique()
  c = 0
  for i in tqdm(range(0,len(df))):
    x = df.iloc[i]
    if x['trunk_road_flag'] == missing:
      y = df[(df.trunk_road_flag!='Data missing or out of range') &(df.first_road_number==x['first_road_number'])].trunk_road_flag.unique()
      if len(y)>0:
        max = 0
        s = ""
        for v in y:
          z = df[(df.trunk_road_flag==v) &(df.first_road_number==x['first_road_number'])]
          if len(z)>max:
            max = len(z)
            s = v
        if s=="":
          print("balabezo")
        res.append(s)
      else:
        temp = df[df.first_road_class==x['first_road_class']]
        y = temp.trunk_road_flag.unique()
        if len(y)>0:
          max = 0
          s =""
          for v in y:
            z = temp[df.trunk_road_flag==v]
            if len(z)>max:
              max = len(z)
              s = v
          if s=="":
            print("balabezo")
          res.append(s)
        else:
          c+=1
          res.append(default)
    else:
      res.append(x['trunk_road_flag'])
  df['trunk_road_flag'] = res

def convert_date_time_week(df):
  result = df.copy()
  day=[]
  month=[]
  week=[]
  hour =[]
  minute= []
  for i in range(0,df.shape[0]):
    date_object = datetime.strptime(result['date'][i],'%d/%m/%Y')
    day.append(date_object.day)
    month.append(date_object.month)
    week.append(date_object.isocalendar()[1])
    time_object = datetime.strptime(result['time'][i],'%H:%M')
    hour.append(time_object.hour)
    minute.append(time_object.minute)
  df['day'] = day
  df['month'] = month
  df['week_number'] = week
  df['hour']=hour
  df['minute'] = minute

def number_encode_features(df,feature,mapping):
    result = df.copy() # take a copy of the dataframe
    map = []
    for x in df[feature]:
      map.append(mapping[x])
    result[feature+'_encoding'] = np.array(map)
    return result

def one_hot_encoding(df,feature,drop_col=None,prefix=None):
  result = pd.get_dummies(df[feature])
  if drop_col!=None:
    result = result.drop([drop_col],axis=1)
  if prefix==None:
    prefix = feature
    
  result = result.add_prefix(prefix+'_')
  res = pd.concat([df, result], axis = 1)
  return res


def calculate_top_categories(df, variable, how_many):
    return [
        x for x in df[variable].value_counts().sort_values(
            ascending=False).head(how_many).index
    ]

def one_hot_encode_frequent(df, variable, how_many):
    result = df.copy()
    top_x_labels = calculate_top_categories(result, variable, how_many)
    for label in top_x_labels:
        result[variable + '_' + label] = np.where(
            result[variable] == label, 1, 0) 
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

def add_one_hot_to_csv(df,feature,drop_col=None,prefix=None):
  cols = df[feature].unique()
  if prefix==None:
    prefix = feature

  for x in cols:
    if x!=drop_col:
      add_to_csv_values(prefix+'_'+x,1,x)

  if drop_col!=None:
    add_to_csv_values(prefix+'_'+drop_col,1,'when all '+feature+'columns 0')

def add_one_hot_freq_to_csv(df, variable, how_many):
    result = df.copy()
    top_x_labels = calculate_top_categories(result, variable, how_many)
    for label in top_x_labels:
      add_to_csv_values(variable + '_' + label,1,label)

def parsing_road(x):
  return int(float(x))
def parsing_LSOA(x):
  return int(float(x[2:]))

def intervals(df,feature,number_of_labels,fun = parsing_road):
  x = df[feature].unique()
  y = []
  for a in x:
    y.append(fun(a))
  y.sort()
  step = (len(y)//number_of_labels)+1
  intervals = [y[0]]
  i = step
  while i<len(y):
    intervals.append(y[i])
    i+=step
  if intervals[-1]==y[-1]:
    intervals[-1] = intervals[-1] +1
  else:
    intervals.append(y[-1]+1)
  return intervals

def discretize(df,feature,intervals,fun= parsing_road,offset=0):
  result = df.copy()
  res = []
  for x in df[feature]:
    x = fun(x)
    for i in range(1,len(intervals)):
      if intervals[i]>x:
        res.append(i-1)
        break
  result[feature+'_encoding'] = np.array(res)
  return result


def discretizeLSOA(df,number_of_labels,add_csv=True):
  result = df.copy()
  res = []
  intervalsW,intervalsE = intervalsLSOA(result,number_of_labels)
  for i in tqdm(range(0,len(result))):
    x = result.iloc[i]['lsoa_of_accident_location']
    if x=='-1':
      res.append('-1')
      continue
    num = int(x[2:])
    if x[0]=='W':
      f = False
      for i in range(1,len(intervalsW)):
        if intervalsW[i]>num:
          res.append(i-1)
          f = True
          break
      if not(f):
        print(x,'W')

    elif x[0]=='E':
      f = False
      for i in range(0,len(intervalsE)):
        if intervalsE[i]>num:
          res.append(i-1+len(intervalsW))
          f = True
          break
      if not(f):
        print(x,'E')
  if add_csv :
    add_intervals_to_csv('lsoa_of_accident_location_encoding',intervalsW,offset = 0 , prefix="W0")
    add_intervals_to_csv('lsoa_of_accident_location_encoding',intervalsE,offset = len(intervalsW) , prefix="E0")
    add_to_csv_values('lsoa_of_accident_location_encoding','-1','NO LSOA ')
    add_to_col_drop('lsoa_of_accident_location')
  result['lsoa_of_accident_location_encoding'] = np.array(res)
  return result

def intervalsLSOA(df,number_of_labels):
  w = []
  e = []
  for x in df.lsoa_of_accident_location:
    if x=='-1':
      continue
    num = int(x[2:])
    if (x[0]=='W') &(not(num in w)):
      w.append(num)
    elif (x[0]=='E') &(not(num in e)):
      e.append(num)
    elif (x[0]!='W') & (x[0]!='E'):
      print(w)
  w.sort()
  e.sort()
  intervalsW = getInterval(w,number_of_labels)
  intervalsE = getInterval(e,number_of_labels)
  return intervalsW,intervalsE

def getInterval(y,number_of_labels):
  step = (len(y)//number_of_labels)+1
  intervals = [y[0]]
  i = step
  while i<len(y):
    intervals.append(y[i])
    i+=step
  if intervals[-1]==y[-1]:
    intervals[-1] = intervals[-1] +1
  else:
    intervals.append(y[-1]+1)
  return intervals

def add_intervals_to_csv(feature,intervals,offset = 0 , prefix=""):
  for i in range(0,len(intervals)-1):
    add_to_csv_values(feature+'_encoding',str(int(i+offset)),feature+" in intrval ["+prefix+str(intervals[i])+" , "+prefix+str(intervals[i+1])+"[")



Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

def get_season(now):
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)
    
def Box_Cox(df,feature):
  index_of_positive = df[feature] > 0
  positive_rows = df[feature].loc[index_of_positive]
  normalized = stats.boxcox(positive_rows)[0]
  return normalized


def outliersLOF(df,featureX,featureY):
  clf = LocalOutlierFactor()
  X = df[[featureX,featureY]].values
  y_pred = clf.fit_predict(X)


  in_mask = [True if l == 1 else False for l in y_pred]
  out_mask = [True if l == -1 else False for l in y_pred]

  c = 0
  for b in out_mask:
    if b :
      c+=1
  return in_mask,c

def outliersBox(df,feature):
  Q1 = df[feature].quantile(0.25)
  Q3 = df[feature].quantile(0.75)
  IQR = Q3 - Q1
  cut_off = IQR * 1.5
  lower = Q1 - cut_off
  upper =  Q3 + cut_off
  in_mask = (df[feature] < upper) & (df[feature] > lower)
  c = 0
  for b in in_mask:
    if not(b) :
      c+=1
  return in_mask,c

def outlierZScore(df,feature,th=3):
  z = np.abs(stats.zscore(df[feature]))
  in_mask = z < th*z.std() + z.mean()
  c = 0
  for b in in_mask:
    if not(b) :
      c+=1
  return in_mask,c


def clean (dataset_path,lookup_path,output_path_csv,year,df):
    no_second_road_then_no_number = '-1'
    df.second_road_number= np.where(df.second_road_class == '-1', no_second_road_then_no_number,df.second_road_number)
    add_to_csv_values('second_road_number','-1','no_second_road_then_no_number')

    df.dropna(subset = ['second_road_number'], inplace=True)

    has_no_number = '0'
    df.second_road_number= np.where(df.second_road_number == 'first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ', has_no_number,df.second_road_number)
    df.first_road_number= np.where(df.first_road_number == 'first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ', has_no_number,df.first_road_number)
    add_to_csv_values('second_road_number','0','first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ')
    add_to_csv_values('first_road_number','0','first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ')

    df.dropna(subset = ['road_type'], inplace=True)

    df.dropna(subset = ['weather_conditions'], inplace=True)
    Data_missing_or_out_of_range = 'Data missing or out of range'

    Not_at_junction_then_no_control= 'Not at junction then no control'
    Not_at_junction_or_within_20_metres = 'Not at junction or within 20 metres'
    df.junction_control= np.where((df.junction_detail == Not_at_junction_or_within_20_metres) & (df.junction_control==Data_missing_or_out_of_range), Not_at_junction_then_no_control,df.junction_control)
    add_to_csv_values('junction_control','Not at junction then no control','Not at junction then no control')



    df = df[(df.junction_detail!=Not_at_junction_or_within_20_metres) | (df.junction_control==Not_at_junction_then_no_control)]

    df = df[df.junction_control!=Data_missing_or_out_of_range]

    df = df[df.road_surface_conditions!=Data_missing_or_out_of_range]

    df = df[df.special_conditions_at_site!=Data_missing_or_out_of_range]

    df = df[df.carriageway_hazards!=Data_missing_or_out_of_range]

    trunk_road_flag_missing(df)

    add_to_col_drop('local_authority_ons_district')

    add_to_col_drop('location_easting_osgr')
    add_to_col_drop('location_northing_osgr')


    LOF_in_mask_number_of_vehicles_number_of_casualties,LOF_countOutliers_number_of_vehicles_number_of_casualties = outliersLOF(df,'number_of_vehicles','number_of_casualties')

    df = df[LOF_in_mask_number_of_vehicles_number_of_casualties]


    Z_in_mask_longitude,Z_countOutliers_longitude =outlierZScore(df,'longitude',th=4)

    df = df[Z_in_mask_longitude]


    Box_in_mask_latitude,Box_countOutliers_latitude = outliersBox(df,'latitude')

    df = df[Box_in_mask_latitude]


    df = df.drop_duplicates(subset=['longitude','latitude','date','time'])

    convert_date_time_week(df)

    ### encoding

    mapping_accident_severity={'Slight':0, 'Serious':1, 'Fatal':2}
    df = number_encode_features(df,'accident_severity',mapping_accident_severity)
    add_mapping_to_csv('accident_severity',mapping_accident_severity)
    add_to_col_drop('accident_severity')

    df = one_hot_encoding(df,'day_of_week','Friday')
    add_one_hot_to_csv(df,'day_of_week','Friday')
    add_to_col_drop('day_of_week')

    df = one_hot_encoding(df,'first_road_class','Unclassified')
    add_one_hot_to_csv(df,'first_road_class','Unclassified')
    add_to_col_drop('first_road_class')

    df = one_hot_encoding(df,'road_type','Slip road')
    add_one_hot_to_csv(df,'road_type','Slip road')
    add_to_col_drop('road_type')

    df = one_hot_encode_frequent(df, 'junction_detail', 5)
    add_one_hot_freq_to_csv(df, 'junction_detail', 5)
    add_to_col_drop('junction_detail')

    df = one_hot_encode_frequent(df, 'junction_control', 3)
    add_one_hot_freq_to_csv(df, 'junction_control', 3)
    add_to_col_drop('junction_control')

    df = one_hot_encoding(df,'second_road_class','Unclassified')
    add_one_hot_to_csv(df,'second_road_class','Unclassified')
    add_to_col_drop('second_road_class')

    df = one_hot_encode_frequent(df,'pedestrian_crossing_human_control',1)
    add_one_hot_freq_to_csv(df, 'pedestrian_crossing_human_control', 1)
    add_to_col_drop('pedestrian_crossing_human_control')

    df = one_hot_encode_frequent(df,'pedestrian_crossing_physical_facilities',3)
    add_one_hot_freq_to_csv(df, 'pedestrian_crossing_physical_facilities', 3)
    add_to_col_drop('pedestrian_crossing_physical_facilities')

    mapping_light_conditions = {'Daylight':0, 'Darkness - lighting unknown':1, 'Darkness - lights lit':2,
        'Darkness - lights unlit':3, 'Darkness - no lighting':4}
    df = number_encode_features(df,'light_conditions',mapping_light_conditions)
    add_mapping_to_csv('light_conditions',mapping_light_conditions)
    add_to_col_drop('light_conditions')

    df = one_hot_encode_frequent(df,'weather_conditions',2)
    add_one_hot_freq_to_csv(df,'weather_conditions',2)
    add_to_col_drop('weather_conditions')

    df = one_hot_encode_frequent(df,'road_surface_conditions',2)
    add_one_hot_freq_to_csv(df,'road_surface_conditions',2)
    add_to_col_drop('road_surface_conditions')

    df = one_hot_encode_frequent(df,'special_conditions_at_site',1)
    add_one_hot_freq_to_csv(df,'special_conditions_at_site',1)
    add_to_col_drop('special_conditions_at_site')

    df = one_hot_encode_frequent(df,'carriageway_hazards',1)
    add_one_hot_freq_to_csv(df,'carriageway_hazards',1)
    add_to_col_drop('carriageway_hazards')

    df = one_hot_encoding(df,'urban_or_rural_area','Urban')
    add_one_hot_to_csv(df,'urban_or_rural_area','Urban')
    add_to_col_drop('urban_or_rural_area')

    df = one_hot_encoding(df,'did_police_officer_attend_scene_of_accident','No')
    add_one_hot_to_csv(df,'did_police_officer_attend_scene_of_accident','No')
    add_to_col_drop('did_police_officer_attend_scene_of_accident')

    df = number_encode_features(df,'police_force',generateLabelsMapping(df,'police_force'))
    add_mapping_to_csv('police_force',generateLabelsMapping(df,'police_force'))
    add_to_col_drop('police_force')

    df = one_hot_encoding(df,'trunk_road_flag','Non-trunk')
    add_one_hot_to_csv(df,'trunk_road_flag','Non-trunk')
    add_to_col_drop('trunk_road_flag')

    df = number_encode_features(df,'local_authority_district',generateLabelsMapping(df,'local_authority_district'))
    add_mapping_to_csv('local_authority_district',generateLabelsMapping(df,'local_authority_district'))
    add_to_col_drop('local_authority_district')

    df = number_encode_features(df,'local_authority_highway',generateLabelsMapping(df,'local_authority_highway'))
    add_mapping_to_csv('local_authority_highway',generateLabelsMapping(df,'local_authority_highway'))
    add_to_col_drop('local_authority_highway')

    df = discretize(df,'first_road_number',intervals(df,'first_road_number',15))
    df = discretize(df,'second_road_number',intervals(df,'second_road_number',15))

    add_intervals_to_csv('first_road_number',intervals(df,'first_road_number',15))
    add_intervals_to_csv('second_road_number',intervals(df,'second_road_number',15))

    add_to_col_drop('first_road_number')
    add_to_col_drop('second_road_number')

    df = discretizeLSOA(df,20)

    df = discretize(df,'week_number',intervals(df,'week_number',5))
    add_intervals_to_csv('week_number',intervals(df,'week_number',5))
    add_to_col_drop('week_number')

    accident_in_summer =[]
    for date in df.date:
        season = get_season(datetime.strptime(date,'%d/%m/%Y'))
        val = 0
        if season=='summer':
            val = 1
        accident_in_summer.append(val)
    df['accident_in_summer'] = np.array(accident_in_summer)
    add_to_csv_values('accident_in_summer',1,'accident happend in summer')

    accident_in_PM =[]
    for time in df.time:
        hour = datetime.strptime(time,'%H:%M').hour
        val = 0
        if hour>=12:
            val = 1
        accident_in_PM.append(val)
    df['accident_in_PM'] = np.array(accident_in_PM)
    add_to_csv_values('accident_in_PM',1,'accident happend in PM time')


    accident_on_weekend = []
    for day in df.day_of_week:
        val = 0
        if day == 'Saturday' or day=='Sunday':
            val = 1
        accident_on_weekend.append(val)
    df['accident_on_weekend'] = np.array(accident_on_weekend)
    add_to_csv_values('accident_on_weekend',1,'accident happend on weekend')

    add_to_col_drop('date')
    add_to_col_drop('time')
    add_to_col_drop('accident_year')
    add_to_col_drop('accident_reference')

    df = df.drop(cols_drop_csv,axis=1)
    np.savetxt(lookup_path, values_csv, delimiter=',', fmt=['"%s"' , '"%s"', '"%s"'], header='column,value,meaning', comments='')


    df['number_of_vehicles'] = Box_Cox(df,'number_of_vehicles')

    df['number_of_casualties'] = Box_Cox(df,'number_of_casualties')

    df['speed_limit'] = Box_Cox(df,'speed_limit')


    df.to_csv(output_path_csv,index=True)



def runClean(dataset_path,lookup_path,output_path_csv):
    df= pd.read_csv(dataset_path,index_col=0)
    clean (dataset_path,lookup_path,output_path_csv,2014,df)