import pandas as pd
import mlflow
import mlflow.sklearn
from mp223.train import ROOT, FEATURES, TARGET, MODEL, URI

def main():
    mlflow.set_tracking_uri(URI)
    model = mlflow.sklearn.load_model(f'models:/{MODEL}@champion')
    holdout = pd.read_csv(ROOT / 'data/raw/holdout_features.csv')
    result = pd.DataFrame({'record_id': holdout['record_id'], TARGET: model.predict(holdout[FEATURES])})
    output = ROOT / 'reports/holdout_predictions.csv'
    output.parent.mkdir(exist_ok=True)
    result.to_csv(output, index=False)
    print(f'Saved {len(result)} predictions to {output}')

if __name__ == '__main__':
    main()
