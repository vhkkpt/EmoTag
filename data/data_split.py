import pandas as pd
from sklearn.model_selection import train_test_split

test_ratio=0.2
val_ratio=0.1

df=pd.read_csv('data_auto.csv')

train, temp = train_test_split(df, test_size=test_ratio+val_ratio)
test, val = train_test_split(temp, test_size=val_ratio/(test_ratio+val_ratio))

print('train:', len(train))
print('test:', len(test))
print('val:', len(val))

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
val.to_csv('val.csv', index=False)
