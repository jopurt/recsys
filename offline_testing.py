import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

file_movies = 'movies.dat'
file_ratings = 'ratings.dat'
file_users = 'users.dat'

data = pd.read_csv(file_ratings, delimiter='::', encoding='ISO-8859-1', header=None, engine='python')
data = data.rename(columns={0: 'UserID', 1: 'MovieID', 2: 'Rating', 3: 'Timestamp'})
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')

train_data = data[data['Timestamp'] < '2001-01-02']
test_data = data[data['Timestamp'] >= '2001-01-02']

all_user_ids = pd.concat([train_data['UserID'], test_data['UserID']]).unique()
all_movie_ids = pd.concat([train_data['MovieID'], test_data['MovieID']]).unique()


def create_sparse_matrix(data, all_user_ids, all_movie_ids):
    user_cat = pd.Categorical(data['UserID'], categories=all_user_ids)
    movie_cat = pd.Categorical(data['MovieID'], categories=all_movie_ids)
    rows = user_cat.codes
    cols = movie_cat.codes
    ratings = data['Rating'].values
    sparse_matrix = coo_matrix((ratings, (rows, cols)), shape=(len(all_user_ids), len(all_movie_ids)))
    return sparse_matrix

train_matrix = create_sparse_matrix(train_data, all_user_ids, all_movie_ids)
train_matrix_csr = train_matrix.tocsr()

test_data = test_data[test_data['UserID'].isin(all_user_ids)]
test_data = test_data[test_data['MovieID'].isin(all_movie_ids)]

user_id_to_idx = {user_id: idx for idx, user_id in enumerate(all_user_ids)}
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(all_movie_ids)}
idx_to_movie = {v: k for k, v in movie_id_to_idx.items()}

model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
model.fit(train_matrix_csr.T)

def recommend(user_id, model, user_items, top_k=10):
    user_idx = user_id_to_idx.get(user_id, -1)
    if user_idx == -1:
        print(f"Пользователь {user_id} отсутствует в обучающих данных")
        return []
    try:
        recommendations = model.recommend(user_idx, user_items[user_idx], N=top_k * 2)
    except IndexError as e:
        return []

    recommended_movies = []
    for rec in recommendations:
        movie_idx = rec[0]
        if movie_idx in idx_to_movie:
            recommended_movies.append(idx_to_movie[movie_idx])
        if len(recommended_movies) >= top_k:
            break
    return recommended_movies


def precision_at_k(recommended_items, relevant_items, k):
    recommended_items = recommended_items[:k]
    hits = len(set(recommended_items) & set(relevant_items))
    return hits / k if k > 0 else 0


def recall_at_k(recommended_items, relevant_items, k):
    recommended_items = recommended_items[:k]
    hits = len(set(recommended_items) & set(relevant_items))
    return hits / len(relevant_items) if len(relevant_items) > 0 else 0

def diversity_score(recommended_items, item_features):
    valid_recommendations = [movie for movie in recommended_items if movie in item_features.index]
    if len(valid_recommendations) < 2:
        return 0.0
    features = item_features.loc[valid_recommendations].values
    similarity_matrix = cosine_similarity(features)
    upper_triangle = np.triu(similarity_matrix, k=1)
    return 1 - upper_triangle.sum() / (len(valid_recommendations) * (len(valid_recommendations) - 1) / 2)


def inverse_propensity_scoring(recommended_items, relevant_items, propensity_scores, k):
    recommended_items = recommended_items[:k]
    score = 0
    for item in recommended_items:
        if item in relevant_items:
            score += 1 / propensity_scores.get(item, 1e-6)
    return score / k if k > 0 else 0


# ips
propensity_scores = train_data.groupby('MovieID').size() / len(train_data)
propensity_scores = propensity_scores.to_dict()


def replay_simulation(model, historical_data, user_items):
    total_reward = 0
    for _, row in historical_data.iterrows():
        user_id = row['UserID']
        movie_id = row['MovieID']
        recommended_items = recommend(user_id, model, user_items, top_k=10)
        if movie_id in recommended_items:
            total_reward += 1
    return total_reward / len(historical_data)

movies = pd.read_csv('movies.csv')
movies = movies[movies['movieId'].isin(all_movie_ids)]
movies['Year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)

genres = movies['genres'].str.get_dummies(sep='|')

movie_features = pd.concat([movies['movieId'], genres, movies['Year']], axis=1)
movie_features = movie_features.rename(columns={'movieId': 'MovieID'})

tags = pd.read_csv('tags.csv')
tags = tags[tags['movieId'].isin(all_movie_ids)]

tag_sentences = tags.groupby('movieId')['tag'].apply(lambda x: list(x)).tolist()
word2vec_model = Word2Vec(sentences=tag_sentences, vector_size=100, window=5, min_count=1, workers=4)

def get_movie_embeddings(movie_id, tags, word2vec_model):
    movie_tags = tags[tags['movieId'] == movie_id]['tag']
    if len(movie_tags) == 0:
        return None
    embeddings = [word2vec_model.wv[tag] for tag in movie_tags if tag in word2vec_model.wv]
    if len(embeddings) == 0:
        return None
    return np.mean(embeddings, axis=0)

movie_embeddings = []
for movie_id in all_movie_ids:
    embedding = get_movie_embeddings(movie_id, tags, word2vec_model)
    if embedding is not None:
        movie_embeddings.append([movie_id] + list(embedding))
    else:
        movie_embeddings.append([movie_id] + [0] * word2vec_model.vector_size)

columns = ['MovieID'] + [f'tag_embedding_{i}' for i in range(word2vec_model.vector_size)]
tag_embeddings_df = pd.DataFrame(movie_embeddings, columns=columns)

item_features = pd.merge(movie_features, tag_embeddings_df, on='MovieID', how='left').fillna(0)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(item_features.drop(columns=['MovieID']))
item_features_scaled = pd.DataFrame(scaled_features, columns=item_features.columns[1:])
item_features_scaled.insert(0, 'MovieID', item_features['MovieID'])

item_features_scaled.set_index('MovieID', inplace=True)

metrics = []
test_user_ids = test_data['UserID'].unique()
print(f"Уникальные пользователи в тестовых данных: {len(test_user_ids)}")

for user_id in test_user_ids:
    recommended_movies = recommend(user_id, model, train_matrix_csr, top_k=10)
    relevant_movies = test_data[test_data['UserID'] == user_id]['MovieID'].tolist()

    a = diversity_score(recommended_movies, item_features)
    metrics.append({
        "user_id": user_id,
        "precision@10": precision_at_k(recommended_movies, relevant_movies, 10),
        "recall@10": recall_at_k(recommended_movies, relevant_movies, 10),
        "diversity": a,
        "ips_score": inverse_propensity_scoring(recommended_movies, relevant_movies, propensity_scores, 10)
    })

metrics_df = pd.DataFrame(metrics)

print(metrics_df.mean())
print(len(metrics_df["user_id"]))

replay_score = replay_simulation(model, test_data, train_matrix_csr)
print("Replay Score:", replay_score)