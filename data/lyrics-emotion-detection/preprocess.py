import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('OriginalAnnotations.csv')

new_df = pd.DataFrame()
new_df['content'] = df['lyrics']

positive_scores = {
    'Amazement': 0.8,
    'Calmness': 0.5,
    'Joyful activation': 1.0,
    'Nostalgia': 0.0,
    'Power': 0.2,
    'Sadness': -1.0,
    'Solemnity': 0.0,
    'Tenderness': 0.7,
    'Tension': -0.7
}

cols = positive_scores.keys()
weights = pd.Series(positive_scores)
df['score'] = df[cols].mul(weights).sum(axis=1)

new_df['label'] = df['score'].apply(
    lambda x: 1 if x >= 0 else 0
)

new_df.to_csv('result.csv', index=False)
