import pandas as pd


df = pd.read_csv("Dataset/All_feature_with_interactions.csv")


col = df.pop('target')
df['diagnosis'] = col


df.to_csv("All_feature_with_interactions.csv", index=False)