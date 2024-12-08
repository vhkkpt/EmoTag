import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_excel('PERC_mendelly.xlsx')

new_df = pd.DataFrame()
new_df['content'] = df['Poem']

positive_emotions = ['courage', 'joy', 'love', 'peace', 'surprise']
new_df['label'] = df['Emotion'].apply(
    lambda x: 1 if isinstance(x, str) and x.strip() in positive_emotions else 0
)

train_df, test_df = train_test_split(new_df, test_size=0.1, random_state=42)

print(f'Training set size: {len(train_df)}')
print(f'Test set size: {len(test_df)}')

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
