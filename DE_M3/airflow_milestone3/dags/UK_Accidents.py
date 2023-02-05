from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import pandas as pd
import numpy as np
# For Label Encoding
from sklearn import preprocessing
import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine

from clean import runClean
from dashboard import CreateDash
from extract import runExtract

dataset = '2014_Accidents_UK.csv'


def transform_clean(filename):
    dataset_path = filename
    lookup_path = '/opt/airflow/data/2014_Accidents_UK_look.csv'
    output_path_csv = '/opt/airflow/data/2014_Accidents_UK_clean.csv'
    runClean(dataset_path,lookup_path,output_path_csv)
    print('loaded after cleaning succesfully')


def extract_additional_resources(filename,lookup):
    dataSet_csv = filename
    lookupPath = lookup
    city_csv_path = '/opt/airflow/data/2014_Accidents_UK_clean_city.csv'
    lookup_path = '/opt/airflow/data/2014_Accidents_UK_clean_city_look.csv'
    runExtract(dataSet_csv,lookupPath,city_csv_path,lookup_path)
    print("extract_additional_resources task success")


def create_dashboard(filename):
    CreateDash(filename)


def load_to_postgres(filename,lookup): 
    df = pd.read_csv(filename)
    df_lookup = pd.read_csv(lookup)
    
    engine = create_engine('postgresql://root:root@pgdatabase:5432/UK_Accidents')
    #engine = create_engine('postgresql://root:root@postgres_accidents_datasets-pgdatabase:5432/UK_Accidents')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'UK_Accidents_2014',con = engine,if_exists='replace')
    df_lookup.to_sql(name = 'lookup_table',con = engine,if_exists='replace')

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'UK_Accidents_pipeline',
    default_args=default_args,
    description='UK Accidents pipeline',
)
with DAG(
    dag_id = 'UK_Accidents_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['Accidents-pipeline'],
)as dag:
    transform_clean_task= PythonOperator(
        task_id = 'transform_clean',
        python_callable = transform_clean,
        op_kwargs={
            "filename": '/opt/airflow/data/2014_Accidents_UK.csv'
        },
    )
    extract_additional_resources_task= PythonOperator(
        task_id = 'extract_additional_resources',
        python_callable = extract_additional_resources,
        op_kwargs={
            "filename": "/opt/airflow/data/2014_Accidents_UK_clean.csv",
            "lookup":"/opt/airflow/data/2014_Accidents_UK_look.csv"
        },
    )
    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "filename": "/opt/airflow/data/2014_Accidents_UK_clean_city.csv",
            "lookup": "/opt/airflow/data/2014_Accidents_UK_clean_city_look.csv"
        },
    )
    create_dashboard_task= PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = create_dashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/2014_Accidents_UK_clean_city.csv"
        },
    )
    

    transform_clean_task >>  extract_additional_resources_task >> load_to_postgres_task >>create_dashboard_task

    
    



