import math
import argparse
import pandas as pd
import mlflow
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(description='Preço imóveis')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.3,
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='profundidade máxima',
    )
    return parser.parse_args()

def main():
    df = pd.read_csv('data/processed/casas.csv')

    X = df.drop('preco', axis=1)
    y = df['preco'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7,)
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('preco-imoveis-script')

    args = parse_args()
    xgb_params = {
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'seed': 7,
    }

    with mlflow.start_run():
        mlflow.xgboost.autolog()
        xgb = xgboost.train(xgb_params, dtrain, evals=[(dtrain, 'train')])
        
        predict = xgb.predict(dtest)
        mse = mean_squared_error(y_test, predict)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, predict)

        mlflow.log_metric('mse', mse)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('r2', r2)

if __name__ == '__main__':
    main()


