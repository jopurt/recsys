import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

file_movies = 'movies.dat'
file_ratings = 'ratings.dat'
file_users = 'users.dat'

df_movies = pd.read_csv(file_movies, delimiter='::', encoding='ISO-8859-1', header=None, engine='python')
df_movies = df_movies.rename(columns={0: 'MovieID', 1: 'Title', 2: 'Genres'})

df_ratings = pd.read_csv(file_ratings, delimiter='::', encoding='ISO-8859-1', header=None, engine='python')
df_ratings = df_ratings.rename(columns={0: 'UserID', 1: 'MovieID', 2: 'Rating', 3: 'Timestamp'})

df_users = pd.read_csv(file_users, delimiter='::', encoding='ISO-8859-1', header=None, engine='python')
df_users = df_users.rename(columns={0: 'UserID', 1: 'Gender', 2: 'Age', 3: 'Occupation', 4: 'Zip-code'})

print(f"{len(df_movies)}")
print(f"{len(df_ratings)}")
print(f"{len(df_users)}")

df_ratings = df_ratings[:5000]
df_users = df_users[:2000]

valid_movie_ids = df_movies['MovieID'].unique()
df_ratings = df_ratings[df_ratings['MovieID'].isin(valid_movie_ids)]

df_ratings['Reward'] = (df_ratings['Rating'] >= 4).astype(int)

num_models = 3
epsilon = 0.1
total_rewards = np.zeros(num_models)
pull_counts = np.zeros(num_models)


def item_based_recommendations(user_id, df_ratings, df_movies, top_n=5):
    user_movies = df_ratings[df_ratings['UserID'] == user_id]
    user_movie_ids = user_movies['MovieID'].unique()

    user_movie_matrix = df_ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

    similarity = cosine_similarity(user_movie_matrix.T)
    similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

    recommended_movies = []
    for movie_id in user_movie_ids:
        if movie_id not in similarity_df.index:
            continue

        similar_movies = similarity_df[movie_id].sort_values(ascending=False).index
        recommended_movies.extend(similar_movies[:top_n])

    return list(set(recommended_movies))


def user_based_recommendations(user_id, df_ratings, df_movies, top_n=5):
    user_movie_matrix = df_ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

    similarity = cosine_similarity(user_movie_matrix)
    similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    similar_users = similarity_df[user_id].sort_values(ascending=False).index[1:top_n + 1]

    recommended_movies = df_ratings[df_ratings['UserID'].isin(similar_users)]['MovieID'].unique()
    return list(recommended_movies)


def content_based_recommendations(user_id, df_ratings, df_movies, top_n=5):
    user_movies = df_ratings[df_ratings['UserID'] == user_id]
    user_movie_ids = user_movies['MovieID'].unique()

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_movies['Genres'])

    similarity = cosine_similarity(tfidf_matrix)
    similarity_df = pd.DataFrame(similarity, index=df_movies['MovieID'], columns=df_movies['MovieID'])

    recommended_movies = []
    for movie_id in user_movie_ids:
        if movie_id not in similarity_df.index:
            continue

        similar_movies = similarity_df.loc[movie_id].sort_values(ascending=False).index
        recommended_movies.extend(similar_movies[:top_n])

    return list(set(recommended_movies))


user_ids = df_users['UserID'].unique()
for i, user_id in enumerate(user_ids):
    # if i % 100 == 0:
    #     print(f"{i}/{len(user_ids)}")

    user_ratings = df_ratings[df_ratings['UserID'] == user_id]

    for _, row in user_ratings.iterrows():
        actual_movie_id = row['MovieID']
        reward = row['Reward']

        if np.random.rand() < epsilon:
            selected_model_id = np.random.randint(num_models)
        else:
            average_rewards = total_rewards / (pull_counts + 1e-6)
            selected_model_id = np.argmax(average_rewards)

        if selected_model_id == 0:
            recommended_movies = item_based_recommendations(user_id, df_ratings, df_movies)
        elif selected_model_id == 1:
            recommended_movies = user_based_recommendations(user_id, df_ratings, df_movies)
        elif selected_model_id == 2:
            recommended_movies = content_based_recommendations(user_id, df_ratings, df_movies)

        if actual_movie_id in recommended_movies:
            pull_counts[selected_model_id] += 1
            total_rewards[selected_model_id] += reward


average_rewards = total_rewards / (pull_counts + 1e-6)
best_model_id = np.argmax(average_rewards)

model_names = ["Item-Based", "User-Based", "Content-Based"]
print(f"Лучшая модель: {model_names[best_model_id]}")

cumulative_reward = np.sum(total_rewards)
optimal_reward = np.max(average_rewards)
max_possible_reward = np.sum(pull_counts) * optimal_reward
regret = max_possible_reward - cumulative_reward
print(f"Regret (потери): {regret}")
