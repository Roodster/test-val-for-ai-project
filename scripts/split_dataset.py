import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('./../data/synth_data_for_training.csv')

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2)  # 80% training, 20% testing

# Write the splits to separate CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
