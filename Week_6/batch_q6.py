import os
import sys
import pickle
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 2), dt(1, 10)), # valid
    (1, None, dt(1, 2), dt(1, 10)), # valid
    (1, 2, dt(2, 2), dt(2, 3)), # valid
    (None, 1, dt(1, 2, 0), dt(1, 2, 50)), # invalid, duration is 50 seconds
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)), # invalid, duration is 59 seconds 
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)) # invalid, duration is 3601 seconds (1 hour and 1 second)
]

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def read_data(filename, categorical):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)

    return prepare_data(df, categorical)

def save_data(df, filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)
    else:
        df.to_parquet(filename, engine='pyarrow', index=False)

def write_data(filename, df):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)
    else:
        df.to_parquet(filename, engine='pyarrow', index=False)

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)

def download_from_internet(url, local_file_name):
    response = requests.get(url, stream=True)
    print("Status code:", response.status_code)
    print("Content type:", response.headers['content-type'])

    with open(local_file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)

    return local_file_name

def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3', endpoint_url=S3_ENDPOINT_URL)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def save_data_to_local_file(data, filename):
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    df.to_parquet(filename, engine='pyarrow')
    
def main(year, month):
    output_file = get_output_path(year, month)

    local_filename = f"{year:04d}-{month:02d}.parquet"
    save_data_to_local_file(data, local_filename)
    print(local_filename)
    
    # Print the file size
    file_size = os.path.getsize(local_filename)
    print(f"The file size is {file_size} bytes")
    
    s3_file_path = f"{year:04d}-{month:02d}.parquet"
    upload_to_s3(local_filename, "qfive", f"{year:04d}-{month:02d}.parquet")

    input_file_s3 = f"s3://qfive/{s3_file_path}"
    print(input_file_s3)
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file_s3, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    
    sum_of_predicted_durations = df_result['predicted_duration'].sum()
    print(f"The sum of predicted durations is {sum_of_predicted_durations}")

    write_data(output_file, df_result)
    
if __name__ == '__main__': 
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)