import pandas as pd
from sklearn.model_selection import train_test_split

test_ratio = 0.2
val_ratio = 0.2

df = pd.read_csv('data/data_auto.csv')

# Split into train + validation and test
train_val, test = train_test_split(df, test_size=test_ratio, random_state=42)
# Split train_val into train and validation
train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)

print(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
val.to_csv('val.csv', index=False)
