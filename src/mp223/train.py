from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from mp223.data import clean_train_data

ROOT = Path(__file__).resolve().parents[2]
NUM = ['rainfall_mm', 'fertilizer_kg', 'sowing_delay_days', 'irrigation_count', 'soil_ph', 'farm_acres']
CAT = ['seed_variety', 'tehsil']
FEATURES = NUM + CAT
TARGET = 'yield_qtl_per_acre'
MODEL = 'mp223-wheat-yield-model'
URI = f'sqlite:///{ROOT / "mlflow.db"}'

def main():
    mlflow.set_tracking_uri(URI)
    mlflow.set_experiment('mp223-wheat-yield')
    raw = pd.read_csv(ROOT / 'data/raw/train.csv')
    dates = pd.to_datetime(raw['date'])
    unique_dates = sorted(dates.unique())
    cutoff = unique_dates[int(len(unique_dates) * 0.8)]
    train = clean_train_data(raw.loc[dates < cutoff])
    val = clean_train_data(raw.loc[dates >= cutoff])
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
    }
    best = (float('inf'), None, None)
    for name, estimator in models.items():
        pre = ColumnTransformer([
            ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUM),
            ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CAT),
        ])
        pipe = Pipeline([('preprocess', pre), ('model', estimator)])
        with mlflow.start_run(run_name=name) as run:
            pipe.fit(train[FEATURES], train[TARGET])
            pred = pipe.predict(val[FEATURES])
            mae = mean_absolute_error(val[TARGET], pred)
            mlflow.log_params({'model_type': name, 'validation_cutoff': str(cutoff), **estimator.get_params()})
            mlflow.log_metrics({'mae': mae, 'rmse': root_mean_squared_error(val[TARGET], pred), 'r2': r2_score(val[TARGET], pred)})
            info = mlflow.sklearn.log_model(pipe, name='model', input_example=train[FEATURES].head(3))
            print(name, 'MAE:', mae)
            if mae < best[0]:
                best = (mae, run.info.run_id, info.model_uri)
    version = mlflow.register_model(best[2], MODEL)
    MlflowClient().set_registered_model_alias(MODEL, 'champion', version.version)
    print('Best run:', best[1], 'Champion version:', version.version, 'MAE:', best[0])

if __name__ == '__main__':
    main()
