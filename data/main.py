from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///disaster.db')
df = pd.read_sql('SELECT * FROM disaster', engine)
print(df.head())
