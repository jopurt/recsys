import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import TruncatedSVD

file_movies = 'movies.dat'
file_ratings = 'ratings.dat'
file_users = 'users.dat'

df_movies = pd.read_csv(file_movies, delimiter='::', encoding='ISO-8859-1', header=None, engine='python')
df_movies = df_movies.rename(columns={0: 'MovieID', 1: 'Title', 2: 'Genres'})

df_ratings = pd.read_csv(file_ratings, delimiter='::', encoding='ISO-8859-1', header=None, engine='python')
df_ratings = df_ratings.rename(columns={0: 'UserID', 1: 'MovieID', 2: 'Rating', 3: 'Timestamp'})

df_users = pd.read_csv(file_users, delimiter='::', encoding='ISO-8859-1', header=None, engine='python')
df_users = df_users.rename(columns={0: 'UserID', 1: 'Gender', 2: 'Age', 3: 'Occupation', 4: 'Zip-code'})

user_item_matrix = df_ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

item_item_matrix = user_item_matrix.T

item_similarity = cosine_similarity(item_item_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=item_item_matrix.index, columns=item_item_matrix.index)

# --- User-Based ---
def get_user_based_predictions(user_id, user_item_matrix, k):

    similar_users = user_similarity_df[user_id].drop(index=user_id).nlargest(k)
    similar_user_ratings = user_item_matrix.loc[similar_users.index]
    predicted_ratings = (similar_users @ similar_user_ratings) / similar_users.sum()
    return predicted_ratings.dropna()

# --- Item-Based ---
def get_item_based_predictions(user_id, user_item_matrix, n):
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index

    predicted_ratings = {}
    for movie_id in user_item_matrix.columns:
        if movie_id in rated_movies:
            continue
        similar_movies = item_similarity_df[movie_id].drop(index=movie_id).nlargest(n)
        weighted_sum = (similar_movies * user_ratings[similar_movies.index]).sum()
        predicted_ratings[movie_id] = weighted_sum / similar_movies.sum() if similar_movies.sum() > 0 else 0

    return pd.Series(predicted_ratings).dropna()

# --- SVD-Based ---
# R=U * Σ * V^T

def get_svd_predictions(user_id, user_item_matrix, n_components=20):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)  # U * Σ
    item_factors = svd.components_  # V^T

    reconstructed_matrix = np.dot(user_factors, item_factors)

    predicted_ratings = pd.DataFrame(reconstructed_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

    print(predicted_ratings.loc[user_id].dropna())
    exit()
    return predicted_ratings.loc[user_id].dropna()

def calculate_mae(predictions, test_set):
    test_ratings = test_set.set_index("MovieID")["Rating"]
    merged = predictions.to_frame(name="Predicted").join(test_ratings, how="inner")
    return mean_absolute_error(merged["Rating"], merged["Predicted"]) if not merged.empty else None

def calculate_recall_at_k(predictions, test_set, k):
    test_high_rated = set(test_set[test_set["Rating"] >= 4]["MovieID"])
    recommended = set(predictions.nlargest(k).index)
    relevant_recommendations = recommended & test_high_rated
    return len(relevant_recommendations) / len(test_high_rated) if test_high_rated else 0

K_VALUES = [15, 20]
N_VALUES = [35]
N_COMPONENTS_SVD = [20, 50]
users_to_test = range(10, 30)

for k in K_VALUES:
    for n in N_VALUES:
        for n_components in N_COMPONENTS_SVD:
            print(f"\n--- k={k}, n={n}, SVD components={n_components} ---")

            for user_id in users_to_test:
                train_set, test_set = train_test_split(
                    df_ratings[df_ratings["UserID"] == user_id], test_size=0.2, random_state=42
                )
                train_ratings = df_ratings[~df_ratings.index.isin(test_set.index)]
                user_item_matrix_train = train_ratings.pivot_table(
                    index='UserID', columns='MovieID', values='Rating'
                ).fillna(0)

                user_predictions = get_user_based_predictions(user_id, user_item_matrix_train, k)
                mae_user = calculate_mae(user_predictions, test_set)
                recall_user = calculate_recall_at_k(user_predictions, test_set, 10)
                print(f"User {user_id}, MAE (User-Based): {mae_user:.4f}" if mae_user else f"User {user_id}, No User-Based MAE")
                print(f"User {user_id}, Recall@10 (User-Based): {recall_user:.2f}")
                print('---------------------------------------------------------------------------------------------------------')

                item_predictions = get_item_based_predictions(user_id, user_item_matrix_train, n)
                mae_item = calculate_mae(item_predictions, test_set)
                recall_item = calculate_recall_at_k(item_predictions, test_set, 10)
                print(f"User {user_id}, MAE (Item-Based): {mae_item:.4f}" if mae_item else f"User {user_id}, No Item-Based MAE")
                print(f"User {user_id}, Recall@10 (Item-Based): {recall_item:.2f}")
                print('---------------------------------------------------------------------------------------------------------')

                svd_predictions = get_svd_predictions(user_id, user_item_matrix_train, n_components)
                mae_svd = calculate_mae(svd_predictions, test_set)
                recall_svd = calculate_recall_at_k(svd_predictions, test_set, 10)
                print(f"User {user_id}, MAE (SVD): {mae_svd:.4f}" if mae_svd else f"User {user_id}, No SVD MAE")
                print(f"User {user_id}, Recall@10 (SVD): {recall_svd:.2f}")
                print('*********************************************************************************************************\n\n')

