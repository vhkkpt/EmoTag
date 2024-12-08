import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('OriginalAnnotations.csv')

new_df = pd.DataFrame()
new_df['content'] = df['lyrics']

emotions = ['Amazement', 'Calmness', 'Joyful activation', 'Nostalgia', 'Power', 'Sadness', 'Solemnity', 'Tenderness', 'Tension']

for emotion in emotions:
    new_df[emotion] = 0
    new_df.loc[df[emotion] != 0, emotion] = 1

train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)

print(f'Training set size: {len(train_df)}')
print(f'Test set size: {len(test_df)}')

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
