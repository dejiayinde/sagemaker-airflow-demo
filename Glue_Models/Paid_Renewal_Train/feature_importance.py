from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import time
import ast
import sys
import os
import shutil
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

from pyspark.ml.classification import GBTClassifier


s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
s3_resource = boto3.resource('s3')



def main():        
    args = getResolvedOptions(sys.argv, ['JOB_NAME',
                                         'env',
                                         'S3_BUCKET_PREFIX',
                                         'KEEP_COLS',
                                         'S3_PREFIX',
                                         'LABEL_COL',
                                        ])
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    env = args['env']
    env = env.lower()
    bucket_prefix = args['S3_BUCKET_PREFIX']
    if env == 'prod':
        bucket = bucket_prefix
    else:
        bucket = f'{bucket_prefix}-{env}'
    print("BUCKET:: ", bucket)
    keep_cols = ast.literal_eval(args['KEEP_COLS'])
    s3_prefix = args['S3_PREFIX']
    label_col = args['LABEL_COL']

        
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class","org.apache.hadoop.mapred.FileOutputCommitter")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3.enableServerSideEncryption", "true")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3.serverSideEncryptionAlgorithm","AES256")
    spark.sparkContext._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    
    
    # read schema from correlation step
    schema_s3_path = s3_prefix + '/staging/schema/correlation_schema.json'
    obj = s3_resource.Object(bucket, schema_s3_path)
    schema = obj.get()['Body'].read().decode("utf-8")
    schema_dict = eval(schema)['input']
    
    # convert schema to StructType
    schemaFields = []
    for s in schema_dict:
        schemaFields.append(StructField(s['name'], DoubleType(), True))

    schemaStruct = StructType(schemaFields)
    
    
    # downloading the data from S3 into a Dataframe
    s3_train_input = 's3://{}/{}/staging/train/correlation/*'.format(bucket, s3_prefix)
    train_df = spark.read.csv(s3_train_input, inferSchema=False, header=False, schema=schemaStruct) 
    
    s3_test_input = 's3://{}/{}/staging/test/correlation/*'.format(bucket, s3_prefix)
    test_df = spark.read.csv(s3_test_input, inferSchema=False, header=False, schema=schemaStruct)
    
    # vector-assembler will bring all the features to a 1D vector for us to save easily into CSV format
    assembler_cols = train_df.columns
    assembler_cols.remove(label_col)
    
    # create a vector for all the features
    assembler = VectorAssembler(inputCols=assembler_cols, outputCol="features")
    feature_df = assembler.transform(train_df)
    
    # fit the gradient boosted model
    gbt = GBTClassifier(maxIter=2, maxBins=16, maxDepth=20, labelCol=label_col, seed=42)
    model = gbt.fit(feature_df)
    
    # idenfity features to be dropped (feature importance = 0)
    feature_importance_arr = model.featureImportances.toArray()
    drop_indicies = np.where(feature_importance_arr == 0)[0].tolist()
    
    drop_columns = []
    for ind in drop_indicies:
        col = assembler_cols[ind]
        if col not in keep_cols:
            drop_columns.append(assembler_cols[ind])
    
    train_df_out = train_df.drop(*drop_columns)
    test_df_out = test_df.drop(*drop_columns)
    
    
    # Convert the transformed dataframes to RDD to save in CSV format and upload to S3
      # train
    s3_output = 's3://{}/{}/staging/train/feature_importance/'.format(bucket, s3_prefix)
    train_df_out.write.format("csv").mode("overwrite").save(s3_output)

      # test
    s3_output = 's3://{}/{}/staging/test/feature_importance/'.format(bucket, s3_prefix)
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
    s3_client.put_object(Body=schema_json, Bucket=bucket, Key=s3_prefix+'/staging/schema/feature_importance_schema.json',ServerSideEncryption='aws:kms')


if __name__ == "__main__":
    main()