'''module to with braze file to s3'''
import sys
import os
from datetime import datetime
import boto3

#Import pyspark modules
from pyspark.context import SparkContext
'''
#Import glue modules
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext


#Initialize contexts and session
spark_context = SparkContext.getOrCreate()
glue_context = GlueContext(spark_context)
spark_Session = glue_context.spark_session

args = getResolvedOptions(sys.argv,
                           ['JOB_NAME',
                            'env',
                            'BUCKET_PREFIX',
                            'S3_PREFIX',
                            'BUCKET_PREFIX_BRAZE',
                            'S3_PREFIX_BRAZE'])

env = args['env'].lower()
bucket_prefix = args['BUCKET_PREFIX']
s3_prefix = args['S3_PREFIX']

bucket_prefix_braze = args['BUCKET_PREFIX_BRAZE']
s3_prefix_braze = args['S3_PREFIX_BRAZE']

if env == 'prod':
    bucket= bucket_prefix
    bucket_braze= bucket_prefix_braze
else:
    bucket = f'{bucket_prefix}-{env}'
    bucket_braze = f'{bucket_prefix_braze}-{env}'
#utc time
year = datetime.utcnow().strftime("%Y")
month = datetime.utcnow().strftime("%m")
day = datetime.utcnow().strftime("%d")

#s3 directory path
input_data_path = f's3://{bucket}/{s3_prefix}/braze_data'
braze_output_data_path = f's3://{bucket_braze}/{s3_prefix_braze}/yr={year}\
/mo={month}/dt={year}-{month}-{day}'
braze_json_prefix = f'{s3_prefix_braze}/yr={year}/mo={month}/dt={year}-{month}-{day}'

#Logic to write brace data to s3
braze_file = spark_Session.read.csv(input_data_path, header="true", \
                                    inferSchema="true", sep=",").coalesce(1)
braze_file.write.format('json').mode("overwrite").save(braze_output_data_path)       #saves file to s3
'''
#function to parse filename
def parse(list_obj):
    '''function to parse file name from s3'''
    for item in list_obj:
        if item.startswith('part-') and item.endswith('.json'):
            obj = item
        else:
            continue
    return obj
'''
#using boto3 client to list s3 content
s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket=bucket_braze, Prefix=braze_json_prefix)
dir_content = [r['Key'].split('/')[-1] for r in response['Contents']]

saved_filename = parse(dir_content)
os.system(f'aws s3 mv {braze_output_data_path}/{saved_filename}  {braze_output_data_path}/pr.json')
'''