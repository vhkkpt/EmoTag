import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('final_df_emotions(remove-bias).csv')

new_df = pd.DataFrame()
new_df['content'] = df['poem content']

emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

for emotion in emotions:
    new_df[emotion] = 0

for emotion in emotions:
    new_df.loc[df['label'] == emotion, emotion] = 1

train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)

print(f'Training set size: {len(train_df)}')
print(f'Test set size: {len(test_df)}')

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)