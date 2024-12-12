import pandas as pd
from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = '''You are an AI specialized in analyzing the sentiment of poems and song lyrics.'''
USER_PROMPT = '''Analyze the sentiment of the following poem or lyrics. If the sentiment is positive, return 1. If the sentiment is negative, return 0. Do not provide any other text or explanations.
{content}'''

df = pd.read_csv('data_original.csv')
labels = []
client = OpenAI()

for content in tqdm(df['content']):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.5,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(content=content)},
        ]
    )
    label = response.choices[0].message.content.strip()
    labels.append(label)

df['label'] = labels
df[['content', 'label']].to_csv('data_auto.csv', index=False)
