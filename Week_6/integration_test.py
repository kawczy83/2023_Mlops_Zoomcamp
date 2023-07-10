from datetime import datetime
import pandas as pd
from batch import prepare_data
from deepdiff import DeepDiff

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)), # valid
        (1, None, dt(1, 2), dt(1, 10)), # valid
        (1, 2, dt(2, 2), dt(2, 3)), # valid
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)), # invalid, duration is 50 seconds
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)), # invalid, duration is 59 seconds 
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)) # invalid, duration is 3601 seconds (1 hour and 1 second)
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    df = prepare_data(df, ['PULocationID', 'DOLocationID'])
    
    expected_data = [
    ('-1', '-1', dt(1, 2), dt(1, 10), 8.0),
    ('1', '-1', dt(1, 2), dt(1, 10), 8.0),
    ('1', '2', dt(2, 2), dt(2, 3), 1.0)
    ]
    expected_df = pd.DataFrame(expected_data, columns=columns + ['duration'])

    # Convert dataframes to dictionaries for comparison
    actual_dict = df.to_dict()
    expected_dict = expected_df.to_dict()

    diff = DeepDiff(actual_dict, expected_dict, ignore_order=True)
    assert diff == {}, f"Unexpected difference: {diff}"

