import numpy as np
import pandas as pd

# Read data
df_book = pd.read_csv('Data/books_data.csv', usecols=['Title', 'description'])
df_rating = pd.read_csv('Data/Books_rating.csv', usecols=['Title', 'User_id', 'review/score'])

print(df_book.shape)
# print(df_book.columns)
# print(df_book.head(5))
print(df_rating.shape)
# print(df_rating.columns)
# print(df_rating.head(5))
# Drop all rows with na values
df_book = df_book.dropna()
df_rating = df_rating.dropna()

# Merge two table
merged_data = pd.merge(df_book,
                       df_rating,
                       on="Title",
                       how="inner")
print(merged_data.head())
# Add Total Rating column
df_title_rating_count = (merged_data.groupby(['Title'])
            .count()
            .sort_values('review/score', ascending=False)
            .reset_index()
            .rename(columns={'review/score':'TotalRating'}))[['Title', 'TotalRating']]
print(df_title_rating_count.head())
threshold_book = np.quantile(df_title_rating_count['TotalRating'], 0.95)
df_new_book = df_title_rating_count.query('TotalRating >= @threshold_book')
print(df_new_book.shape)
df_merged = pd.merge(df_new_book, merged_data, on='Title', how='inner').drop(['TotalRating'], axis=1)
print(df_merged.head())
print(df_merged.shape)

df_users_rating_count = (df_merged.groupby(['User_id'])
            .count()
            .sort_values('review/score', ascending=False)
            .reset_index()
            .rename(columns={'review/score':'TotalRating'}))[['User_id', 'TotalRating']]

# print(df_users_rating_count.head())
print(df_users_rating_count.shape)
# Remove the users's rating in 75% quantile
threshold_users = np.quantile(df_users_rating_count['TotalRating'], 0.9)
print(threshold_users)
df_new_users = df_users_rating_count.query('TotalRating > @threshold_users')
print(df_new_users.shape)

df_new_merged = pd.merge(df_merged, df_new_users, on='User_id', how='inner').drop(['TotalRating'], axis=1)
print(df_new_merged.head())
df_new_merged.drop_duplicates(['User_id', 'Title'])
print(df_new_merged.shape)

df_new_merged.to_csv('Data/new_book_Rating_095.csv',index=False)
#
# print(df_new_merged.head(5))
