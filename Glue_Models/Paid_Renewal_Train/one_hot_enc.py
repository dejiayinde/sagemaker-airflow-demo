from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import time
import sys
import os
import shutil
import csv
import ast
import boto3
from botocore.client import Config

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

import pyspark
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructField, StructType, StringType, DoubleType
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import *


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
                                         'S3_PREFIX',
                                         'ONEHOT_COLS',
                                         'DROP_COLS',
                                         'LABEL_COL',
                                        ])
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    env = args['env']
    env = env.lower()
    print("ENV:: ", env)
    bucket_prefix = args['S3_BUCKET_PREFIX']
    if env == 'prod':
        bucket = bucket_prefix
    else:
        bucket = f'{bucket_prefix}-{env}'
    print("BUCKET:: ", bucket)
    
    s3_prefix = args['S3_PREFIX']
    onehot_encode_cols = ast.literal_eval(args['ONEHOT_COLS'])
    drop_cols = ast.literal_eval(args['DROP_COLS'])
    label_col = args['LABEL_COL']

    for col in onehot_encode_cols:
        err = "Exception raised: Invalid Column Format for One Hot Columns: Needed <Str>, but found <{}> in Column {}".format(type(col), col)
        if not isinstance(col, str):
            raise Exception(err)

    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class","org.apache.hadoop.mapred.FileOutputCommitter")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3.enableServerSideEncryption", "true")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3.serverSideEncryptionAlgorithm","AES256")
    spark.sparkContext._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    
    # Downloading the data from S3 into a Dataframe
    s3_train_input = 's3://{}/{}/staging/train/raw/*'.format(bucket, s3_prefix)
    train_df = spark.read.csv(s3_train_input, inferSchema=True, header=True) #schema=schema
    
    s3_test_input = 's3://{}/{}/staging/test/raw/*'.format(bucket, s3_prefix)
    test_df = spark.read.csv(s3_test_input, inferSchema=True, header=True) #schema=schema
    
    #drop HBO_UUID column
    for col in drop_cols:
        train_df = train_df.drop(col)
        test_df = test_df.drop(col)
    
    print("DROP SUCCESS")
    
    pipeline_lst = []
    for col in onehot_encode_cols:
        indexer = StringIndexer(inputCol=col, outputCol="indexed_"+col, handleInvalid='keep')
        encoder = OneHotEncoder(inputCol='indexed_'+col, outputCol=col+"_vec", dropLast=True)
        pipeline_lst.append(indexer)
        pipeline_lst.append(encoder)
    
   
    # vector-assembler will bring all the features to a 1D vector for us to save easily into CSV format
    assembler_cols = train_df.columns
    for col in onehot_encode_cols:
        assembler_cols.remove(col)
        assembler_cols.append(col + '_vec')

    # remove target column
    assembler_cols.remove(label_col)

    assembler = VectorAssembler(inputCols=assembler_cols, outputCol="features")

    pipeline_lst.append(assembler)
    pipeline = Pipeline(stages=pipeline_lst)
    try:
        model = pipeline.fit(train_df)
        print("MODEL FIT SUCCESS")
        
        # collect and record the schema
        '''
        start with target column, then the remaining non-one-hot-encoded columns, then the one-hot-encoded columns in order
        '''
        model_columns = [label_col]

        all_input_cols = train_df.columns
        for col in onehot_encode_cols:
            all_input_cols.remove(col)

        # remove target column
        all_input_cols.remove(label_col)
        # add the remaining feature columns
        model_columns.extend(all_input_cols)
        x = 0
        for col in onehot_encode_cols:
            model_columns.extend([col + "_" + str(i) for i in model.stages[x].labels])
            x += 2

        # write the model schema to S3
        schema = {"input":[]
                ,"output":{
                    "name": "features",
                    "type": "double",
                    "struct": "vector"
                    }
                }
        
        for col in model_columns:
            schema["input"].append({
                "name": col,
                "type": "double"
            })

        schema_json = json.dumps(schema)                    

        # write schema to S3
        s3_client.put_object(Body=schema_json, Bucket=bucket, Key= s3_prefix+'/staging/schema/onehotenc_schema.json', ServerSideEncryption='aws:kms')
        

        print("SCHEMA DUMP SUCCESS")
        # This step transforms the datasets with information obtained from the previous fit
        transformed_train = model.transform(train_df)
        transformed_test = model.transform(test_df)
        
    #     label_col = 'FLG_TARGET'
        features_vec_col = 'features'
        def extract(row):
            return (row[label_col], ) + tuple(row[features_vec_col].toArray().tolist())
        
        #Expands the transformed datasets
        transformed_train_expanded = transformed_train.rdd.map(extract).toDF([label_col])
        transformed_test_expanded = transformed_test.rdd.map(extract).toDF([label_col])

        print("TRANSFORMED SUCCESS")

        # Convert the transformed dataframes to RDD to save in CSV format and upload to S3
        # train
        s3_output = 's3://{}/{}/staging/train/onehotenc/'.format(bucket, s3_prefix)
        transformed_train_expanded.write.format("csv").mode("Overwrite").save(s3_output)
        # test
        s3_output = 's3://{}/{}/staging/test/onehotenc/'.format(bucket, s3_prefix)
        transformed_test_expanded.write.format("csv").mode("Overwrite").save(s3_output)

        print("CSV DUMP SUCCESS")

        s3_model_output = 's3://{}/{}/staging/onehotenc_model_output'.format(bucket, s3_prefix)
        model.write().overwrite().save(s3_model_output)

        print("MODEL WRITE SUCCESS")
    except Exception as e:
        print("type of Exception: {}".format(type(e)))
        msg = "one_hot_enc Job Failed: logs returned from job: {}".format(str(e))
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()