import pandas as pd
import sys

def main():
    file_path = sys.argv[1]
    df = pd.read_excel(file_path)
    unique_emotions = sorted(df['Emotion'].dropna().astype(str).unique())
    for emotion in unique_emotions:
        count = df[df['Emotion'] == emotion].shape[0]
        print(f"{emotion}: {count}")

if __name__ == "__main__":
    main()
