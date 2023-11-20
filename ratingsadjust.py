import pandas as pd

# Load the datasets
ratings_df = pd.read_csv('./ml-latest-small/ratings.csv')
tags_df = pd.read_csv('./ml-latest-small/tags.csv')
def adjust_rating(row):
    if row['tag_count'] == 0:
        return row['rating']
    elif row['rating'] >= 3 and row['tag_count'] >= 2:
        return row['rating'] + row['tag_count'] * 0.5
    elif row['rating'] <= 1:
        return 0
    else:
        return row['rating']
# Preview the data
print(ratings_df.head())
print(tags_df.head())
# Counting the number of tags per movie
# Grouping by both userId and movieId to count tags
tag_counts_per_user_movie = tags_df.groupby(['userId', 'movieId']).size()

# Merge the tag count with the ratings dataframe on both userId and movieId
merged_df_corrected = ratings_df.merge(tag_counts_per_user_movie.rename('tag_count'),
                                       on=['userId', 'movieId'],
                                       how='left')

# Fill NaN values in tag_count with 0 (indicating no tags by the user for that movie)
merged_df_corrected['tag_count'] = merged_df_corrected['tag_count'].fillna(0)

# Apply the adjusted rating rules
merged_df_corrected['adjusted_rating'] = merged_df_corrected.apply(adjust_rating, axis=1)
# 保存更新后的 DataFrame 到新的 CSV 文件
# output_csv_path = 'ratingsadjust.csv'
# merged_df_corrected.to_csv(output_csv_path, index=False)
# # Display the first few rows of the adjusted ratings with the corrected approach
# print(merged_df_corrected.head())
# Load the additional datasets
updated_movies_df = pd.read_csv('updated_movies.csv')
ratings_adjust_df = pd.read_csv('ratingsadjust.csv')

# Preview the data
print(updated_movies_df.head())
print(ratings_adjust_df.head())

# Merging the two datasets on movieId
merged_movies_df = updated_movies_df.merge(ratings_adjust_df, on='movieId', how='right')

# Display the first few rows of the merged dataset
merged_movies_df.head()
merged_movies_df.to_csv('merged_ratingsadjust.csv', index=False)
print(merged_movies_df.head())