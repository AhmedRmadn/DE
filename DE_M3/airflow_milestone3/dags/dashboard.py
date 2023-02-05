import plotly.express as px 
from dash import Dash, dcc, html, Input, Output

import pandas as pd
import numpy as np
import math


def mapPlot(df,mapping):
  resDataFrame = pd.DataFrame()
  for feature in mapping.keys():
    tempArray = df[feature].values
    res = []
    map = mapping[feature]
    for i in range(tempArray.shape[0]):
      res.append(map[str(tempArray[i])])
    # for key in map.keys():
    #   mask = tempArray==key
    #   tempArray = np.where(mask, map[key],tempArray)
    resDataFrame[feature] = res
  return resDataFrame

def plot1(df):
    map1={"accident_severity_encoding":{'0': 'Slight', '1': 'Serious', '2': 'Fatal'},'did_police_officer_attend_scene_of_accident_Yes':{'1':'Yes','0':'No'}}
    dfPlot1 = mapPlot(df,map1)


    fig = px.histogram(dfPlot1, x="accident_severity_encoding",
                color='did_police_officer_attend_scene_of_accident_Yes', barmode='group',
                histfunc='count',log_y=True,labels={'did_police_officer_attend_scene_of_accident_Yes':'police_attend'},
                height=600,title="accident severity vs police attend the scene of the accident log y")
    return fig

def plot2(df):
    map2={"urban_or_rural_area_Rural":{'1': 'Rural', '0': 'Urban'},'did_police_officer_attend_scene_of_accident_Yes':{'1':'Yes','0':'No'}}
    dfPlot2 = mapPlot(df[df["accident_severity_encoding"]==2],map2)

    fig = px.histogram(dfPlot2, x="urban_or_rural_area_Rural",
                color='did_police_officer_attend_scene_of_accident_Yes', barmode='group',
                histfunc='count',log_y=True,labels={'did_police_officer_attend_scene_of_accident_Yes':'police_attend',"accident_severity_encoding":"accident_severity"},
                height=600,title="when accident severity Fatel, area type vs police attend the scene of the accident log y")
    return fig

def plot3(df):
    map3={"accident_severity_encoding":{'0': 'Slight', '1': 'Serious', '2': 'Fatal'},"urban_or_rural_area_Rural":{'1': 'Rural', '0': 'Urban'}}
    dfPlot3 = mapPlot(df,map3)

    fig = px.histogram(dfPlot3, x="accident_severity_encoding",
                color='urban_or_rural_area_Rural', barmode='group',
                histfunc='count',log_y=True,labels={'urban_or_rural_area_Rural':'urban or Rural','accident_severity_encoding':'accident_severity'},
                height=600,title="accident severity vs  area type log y")
    return fig

def plot4(df):
    map4={"light_conditions_encoding":{'0': 'Daylight', '1': 'Darkness - lighting unknown', '2': 'Darkness - lights lit','3':'Darkness - lights unlit','4':'Darkness - no lighting'},"urban_or_rural_area_Rural":{'1': 'Rural', '0': 'Urban'}}
    dfPlot4 = mapPlot(df,map4)

    fig = px.histogram(dfPlot4, x="urban_or_rural_area_Rural",
                color='light_conditions_encoding', barmode='group',
                histfunc='count',log_y=True,labels={'urban_or_rural_area_Rural':'urban or Rural','light_conditions_encoding':'light_conditions'},
                height=600,title="light conditions vs area type log y")
    return fig

def plot5(df):
    map5={"light_conditions_encoding":{'0': 'Daylight', '1': 'Darkness - lighting unknown', '2': 'Darkness - lights lit','3':'Darkness - lights unlit','4':'Darkness - no lighting'},"accident_severity_encoding":{'0': 'Slight', '1': 'Serious', '2': 'Fatal'}}
    dfPlot5 = mapPlot(df,map5)

    fig = px.histogram(dfPlot5, x="accident_severity_encoding",
                color='light_conditions_encoding', barmode='group',
                histfunc='count',log_y=True,labels={'accident_severity_encoding':'accident_severity','light_conditions_encoding':'light_conditions'},
                height=600,title="accident severity  vs light_conditions log y")
    return fig

def CreateDash(dataset_path):
    df= pd.read_csv(dataset_path,index_col=0)
    app = Dash()
    app.layout = html.Div([
        html.H1("UK_Accidents_2014", style={'text-align': 'center'}),
        html.Br(),
        html.Div(),
        dcc.Graph(figure=plot1(df)),
        html.Br(),
        html.Div(),
        dcc.Graph(figure=plot2(df)),
        html.Br(),
        html.Div(),
        dcc.Graph(figure=plot3(df)),
        html.Br(),
        html.Div(),
        dcc.Graph(figure=plot4(df)),
        html.Br(),
        html.Div(),
        dcc.Graph(figure=plot5(df)),
    ])

    app.run_server(host ='0.0.0.0',debug=False)
    #app.run_server(debug=False,port='8002')