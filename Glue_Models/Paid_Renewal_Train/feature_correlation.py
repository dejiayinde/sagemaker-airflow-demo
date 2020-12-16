"""
Parameters Format (Example):
{
"S3_BUCKET" {AWS S3 BUCKET}: (EXAMPLE) "hbomax-datascience-development-dev",
"KEEP_COLS" {LIST OF COLUMNS YOU WANT TO KEEP AFTER CORRELATION (STRING FORMAT)}: (EXAMPLE) "['NUM_NOT_ENGAGED_STREAM' 
        ,'MOBILE_PERCENT_ADJ' 
        ,'LG_SCRN_PERCENT_ADJ'
        ,'NUM_SERIES_LOYAL_W_EPISODE_ON_AIR_BILLINGDT'
        ,'NUM_SERIES_LOYAL_INLIB_LATESTSEASON'
        ,'NUM_SERIES_LOYAL'
        ,'AVG_SERIES_COMPLETION_RATE_MAXEPISODE'
        ,'FLAG_SERIES_ENGAGED_FT']",
"S3_PREFIX" {FILE PATH IN S3 BUCKET} : (EXAMPLE) "lifecycle/paid-renewal-daily-model",
"IGNORE_COLS" {TARGET INDICATOR COLUMNS (STRING FORMAT)}: (EXAMPLE) "['FLG_TARGET']",
"CORR_TRESHOLD" {CORRELATION TRESHOLD VALUE} (STRING FORMAT): (EXAMPLE) "0.75"
}
"""

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import time
import sys
import os
import shutil
import ast
import csv
import boto3
from botocore.client import Config
import numpy as np

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

import pyspark
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructField, StringType, DoubleType, IntegerType, StructType
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import *
#from mleap.pyspark.spark_support import SimpleSparkSerializer

from pyspark.ml.stat import Correlation

s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
s3_resource = boto3.resource('s3')

def main():        
    '''
    # Define variables
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--S3_BUCKET", type=str, help="s3 input bucket")
    args = parser.parse_args()
    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    bucket = args.S3_BUCKET
    '''
    
    args = getResolvedOptions(sys.argv, ['JOB_NAME',
                                         'env',
                                         'S3_BUCKET_PREFIX',
                                         'KEEP_COLS',
                                         'S3_PREFIX',
                                         'IGNORE_COLS',
                                         'CORR_TRESHOLD',
                                        ])
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    print("ARGS::: ", args)
    env = args['env']
    env = env.lower()
    bucket_prefix = args['S3_BUCKET_PREFIX']
    if env == 'prod':
        bucket = bucket_prefix
    else:
        bucket = f'{bucket_prefix}-{env}'
    print("BUCKET:: ", bucket)
    keep_cols = ast.literal_eval(args["KEEP_COLS"])
    print("KEEP COLS::: ", keep_cols)
    s3_prefix = args["S3_PREFIX"]
    print("S3 PREFIX::: ", s3_prefix)
    ignore_cols = ast.literal_eval(args["IGNORE_COLS"])
    print("IGNORE COLS::: ", ignore_cols)
    corr_treshold = float(args["CORR_TRESHOLD"])
    print("CORR TRESHOLD::: ", corr_treshold)
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class","org.apache.hadoop.mapred.FileOutputCommitter")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3.enableServerSideEncryption", "true")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3.serverSideEncryptionAlgorithm","AES256")
    spark.sparkContext._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    
    # read schema from onehotenc step
    schema_s3_path = s3_prefix + '/staging/schema/onehotenc_schema.json'
    obj = s3_resource.Object(bucket, schema_s3_path)
    schema = obj.get()['Body'].read().decode("utf-8")
    schema_dict = eval(schema)['input']
    
    # convert schema to StructType
    schemaFields = []
    for s in schema_dict:
        schemaFields.append(StructField(s['name'], DoubleType(), True))

    schemaStruct = StructType(schemaFields)
    
    # downloading the data from S3 into a Dataframe
    s3_train_input = 's3://{}/{}/staging/train/onehotenc/*'.format(bucket, s3_prefix)
    train_df = spark.read.csv(s3_train_input, inferSchema=False, header=False, schema=schemaStruct)
    
    s3_test_input = 's3://{}/{}/staging/test/onehotenc/*'.format(bucket, s3_prefix)
    test_df = spark.read.csv(s3_test_input, inferSchema=False, header=False, schema=schemaStruct)
    
    # CREATE CORRELATION MATRIX
    # collect the dimensions of the correlation matrix
    ignore = ignore_cols
    len_ignore = len(ignore)

    num_features = len(train_df.dtypes) - len_ignore
    corr_dims = (num_features, num_features)
    
    # transform features to vectors
    assembler = VectorAssembler(
        inputCols=[x for x in train_df.columns if x not in ignore],
        outputCol='features')
    
    df_feats = assembler.transform(train_df)
    
    # Compute Pearson's r values
    corr = Correlation.corr(df_feats, "features")
    
    # FILTER MATRIX
    # convert matrix results to numpy
    df_corr = np.reshape(corr.collect()[0]["pearson({})".format("features")].values, corr_dims)
    
    # Ignore diagonal results and evaluate against threshold
    mask = ~np.eye(len(df_corr), dtype=bool)
    masked_array = np.ma.array(df_corr, mask=mask).mask * df_corr
    #     mask = np.zeros_like(df_corr, dtype=bool)  # Lower triangle mask
    #     df_corr[np.tril_indices_from(mask)] = False  # Lower triangle mask
    corr_bool = np.absolute(df_corr) > float(corr_treshold)
    
    # Structure correlation matches into dictionary key-value pairs
    input_columns = train_df.columns
    corr_cols = {}
    for i, row in enumerate(corr_bool):
        if row.any():
            corr_cols[input_columns[i+1]] = [input_columns[c+1] for c in np.where(row==True)[0]]
            
    # Scan over corr_cols and identify columns to drop
    drop_list = []
    safe_list = list(keep_cols)
    for k, v in corr_cols.items():
        if k not in safe_list:
            safe = True  # Assume safe until proven otherwise
            for c in v:
                if c in safe_list:
                    safe = False
            if safe:
                safe_list.append(k)
            else:
                drop_list.append(k)
    
    train_df_out = train_df.drop(*drop_list)
    test_df_out = test_df.drop(*drop_list)
    
    
    # Convert the transformed dataframes to RDD to save in CSV format and upload to S3
      # train
    s3_output = 's3://{}/{}/staging/train/correlation/'.format(bucket, s3_prefix)
    train_df_out.write.format("csv").mode("overwrite").save(s3_output)
      # test
    s3_output = 's3://{}/{}/staging/test/correlation/'.format(bucket, s3_prefix)
    test_df_out.write.format("csv").mode("overwrite").save(s3_output)

    # write the sparkml schema to s3
    schema = {"input":[]
              ,"output":{
                "name": "features",
                "type": "double",
                "struct": "vector"
                }
             }
    for col in train_df_out.dtypes:
        #if col[0] != 'FLG_TARGET': # exlcude target attribute
        schema["input"].append({
                "name": col[0],
                "type": col[1]
            })
    
    schema_json = json.dumps(schema)
    s3_correlation_path = s3_prefix + '/staging/schema/correlation_schema.json'
    s3_client.put_object(Body=schema_json, Bucket=bucket, Key=s3_correlation_path,ServerSideEncryption='aws:kms')

if __name__ == "__main__":
    main()