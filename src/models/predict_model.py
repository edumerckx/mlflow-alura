import mlflow
logged_model = 'models:/PrecoImoveis/Production'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

df = pd.read_csv('data/processed/casas_X.csv')

df['predicted'] = loaded_model.predict(pd.DataFrame(df))

df.to_csv('data/processed/predicted.csv', index=False)