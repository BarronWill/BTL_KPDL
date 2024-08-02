import numpy as np
import pandas as pd


df_book = pd.read_csv('Data/books_data.csv', usecols=['Title', 'description'])
df_rating = pd.read_csv('Data/Books_rating.csv', usecols=['Title', 'User_id', 'review/score'])

df_book = df_book.dropna()
merged_data = pd.merge(df_book, df_rating, on="Title", how="inner")
df_count = (merged_data.groupby(['Title'])
            .count()
            .sort_values('review/score', ascending=False)
            .reset_index()
            .rename(columns={'review/score':'TotalRating'}))

#Check null
merged_data = merged_data.dropna()
print(merged_data[merged_data['TotalRating'] > 50])
print(merged_data.columns[merged_data.isnull().any()])
print(df_count['TotalRating'].value_counts())
# merged_data.to_csv('bookRating.csv',index=False)
# print(merged_data.shape)
