import pandas as pd
import pathlib
import yaml
import sys
import joblib
import xgboost



def load_data(train_path):
    train = pd.read_csv(train_path)
    return train

def split_features_traget(data):
    traget = 'Class'
    X = data.drop(columns=traget)
    y = data[traget]
    return X, y

def model(params):
    model = xgboost.XGBClassifier(
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        
    )


def train_model(model, X, y):
    model.fit(X, y)
    return model

def save_model(model, output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path + 'xboost_model.joblib')


def main():
    curr_dir = pathlib.Path(__file__)
    home_dirr = curr_dir.parent.parent.parent
    train_data_file = sys.argv[1]
    train_data_path = home_dirr.as_posix() + train_data_file

    model_output_file = sys.argv[2]
    model_output_path = home_dirr.as_posix() + model_output_file

    params_file = sys.argv[3]
    params_path = home_dirr.as_posix() + params_file
    params_path_file = yaml.safe_load(open(params_path))['train_model']

    data = load_data(train_path=train_data_path)
    X, y = split_features_traget(data)

    model = model(params_path_file)

    train_model = train_model(model=model , X=X, y=y)

    save_model(model = train_model, output_path=model_output_path)




if __name__ == "__main__":
    main()
