import pathlib
import pandas as pd
import sys
import yaml 

from sklearn.model_selection import train_test_split

# create dataset

def data_set(data_path):
    df = pd.read_csv(data_path)
    return df


# split data

def split_data(data, test_size, seed):
    train, test = train_test_split(data, test_size=test_size,random_state=seed)
    return train, test


# save data
def save_data(train, test, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    train.to_csv(path + '/train.csv', index=False)
    test.to_csv(path + '/test.csv', index=False)
# main

def main():
    curr_path = pathlib.Path(__file__)
    home_dirr = curr_path.parent.parent
    input_file = sys.argv[1]
    data_path = home_dirr.as_posix() + input_file
   
    output_path = home_dirr.as_posix() + '/data/processed'

    params_file = home_dirr.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))['make_dataset']


    data = data_set(data_path)
    train, test = split_data(data=data, test_size=params['test_size'], seed=params['seed'])

    save_data(train=train, test=test, path=output_path)






if __name__ == "__main__":
    main()
