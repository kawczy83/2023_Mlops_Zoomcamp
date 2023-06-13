import pickle
import requests
import pandas as pd
import argparse

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(url):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open("/tmp/temp.parquet", 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

        df = pd.read_parquet("/tmp/temp.parquet")
        
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df
    else:
        response.raise_for_status()

def main(year, month):
    df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}-{:02}.parquet'.format(year, month))

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    # Print the mean predicted duration
    print(round(y_pred.mean(),2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Year of the data to process")
    parser.add_argument("--month", type=int, help="Month of the data to process")
    args = parser.parse_args()

    main(args.year, args.month)
