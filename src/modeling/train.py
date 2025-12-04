import pandas as pd
import pathlib
import yaml
import sys
import joblib
import xgboost

from sklearn.model_selection import train_test_split



def load_data(train_path):
    train = pd.read_csv(train_path)
    return train

def split_features_traget(data):
    traget = 'Class'
    X = data.drop(columns=traget)
    y = data[traget]
    return X, y

def models(params):
    model = xgboost.XGBClassifier(
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        early_stopping_rounds=params['early_stopping_rounds'],
        n_jobs=-1
    )
    return model


def train_models(model, X, y,params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=params['random_state'])
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set)
    print(model.n_estimators)
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

    params_path = home_dirr.as_posix() + '/params.yaml'
    params_path_file = yaml.safe_load(open(params_path))

    

    data = load_data(train_path=train_data_path)
    X, y = split_features_traget(data)

    model = models(params=params_path_file['train_model'])

    train_model = train_models(model=model , X=X, y=y, params=params_path_file['train_test_split'])

    save_model(model = train_model, output_path=model_output_path)




if __name__ == "__main__":
    main()
