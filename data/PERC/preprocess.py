import pandas as pd


df = pd.read_excel('PERC_mendelly.xlsx')

new_df = pd.DataFrame()
new_df['content'] = df['Poem']

positive_emotions = ['courage', 'joy', 'love', 'peace', 'surprise']
new_df['label'] = df['Emotion'].apply(
    lambda x: 1 if isinstance(x, str) and x.strip() in positive_emotions else 0
)

new_df.to_csv('result.csv', index=False)
