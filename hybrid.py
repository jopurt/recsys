import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

df = pd.read_csv("T_games_dataset.csv")

columns_to_drop = ['id', 'order_day', 'monthly_income_amt']
df = df.drop(columns=columns_to_drop)

df = df.head(10000)


def collaborative_filtering_model(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['client_id', 'good_id', 'good_cnt']], reader)
    model = SVD()
    trainset = data.build_full_trainset()
    model.fit(trainset)
    return model

model = collaborative_filtering_model(df)

def get_cf_recommendations(user_id, n=10):
    items = df['good_id'].unique()
    predictions = [model.predict(user_id, item) for item in items]
    return sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

def content_based_recommendations(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['good_name'] + " " + df['category_name'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = content_based_recommendations(df)

def get_content_based_recommendations(good_id, n=10):
    idx = df[df['good_id'] == good_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:n + 1]]
    return df.iloc[top_indices][['good_id', 'good_name']].to_dict('records')

def expand_recommendations(user_id, initial_recommendations, n=20):
    user_data = df[df['client_id'] == user_id]
    category = user_data['category_name'].iloc[0]

    same_category = df[df['category_name'] == category]['good_id'].unique().tolist()

    return list(set(initial_recommendations + same_category))[:n]

def filter_recommendations(user_id, expanded_recommendations):
    user_data = df[df['client_id'] == user_id]
    purchased_items = user_data['good_id'].unique()

    filtered = []
    for item in expanded_recommendations:
        item_data = df[df['good_id'] == item]
        if not item_data.empty and item not in purchased_items:
            filtered.append(item)

    return filtered


def sort_recommendations(user_id, filtered_recommendations):
    scores = []
    for item in filtered_recommendations:
        cf_score = model.predict(user_id, item).est
        steam_score = df[df['good_id'] == item]['steam_popularity_score'].iloc[0]

        combined_score = 0.3 * cf_score + 0.7 * steam_score
        scores.append((item, combined_score))

    return [item for item, score in sorted(scores, key=lambda x: x[1], reverse=True)]


def truncate_recommendations(sorted_recommendations, n=10):
    return sorted_recommendations[:n]


def gefest_recommendations(user_id, n=10):
    cf_recs = [rec.iid for rec in get_cf_recommendations(user_id, n=10)]



    content_recs = get_content_based_recommendations(cf_recs[0], n=10)
    initial_recommendations = list(set(cf_recs + [rec['good_id'] for rec in content_recs]))

    expanded_recommendations = expand_recommendations(user_id, initial_recommendations, n=30)

    filtered_recommendations = filter_recommendations(user_id, expanded_recommendations)

    sorted_recommendations = sort_recommendations(user_id, filtered_recommendations)

    final_recommendations = truncate_recommendations(sorted_recommendations, n=n)

    final_recommendations_with_details = []
    total_steam_score = 0
    for good_id in final_recommendations:
        good_name = df[df['good_id'] == good_id]['good_name'].iloc[0]
        steam_score = df[df['good_id'] == good_id]['steam_popularity_score'].iloc[0]
        final_recommendations_with_details.append({
            "good_id": good_id,
            "good_name": good_name,
            "steam_popularity_score": steam_score
        })
        total_steam_score += steam_score

    average_steam_score = total_steam_score / len(final_recommendations) if len(final_recommendations) > 0 else 0

    return final_recommendations_with_details, average_steam_score


user_id = df["client_id"][10]
recommendations, avg_steam_score = gefest_recommendations(user_id, n=10)
for rec in recommendations:
    print(f"ID: {rec['good_id']}, Название: {rec['good_name']}, Steam Popularity Score: {rec['steam_popularity_score']:.2f}")

print(f"\n{avg_steam_score:.2f}")
print(f"{df['steam_popularity_score'].mean():.2f}")

def get_purchased_goods(client_id, df):
    purchased_goods = df[df['client_id'] == client_id]['good_name'].unique()
    return list(purchased_goods)


purchased_goods = get_purchased_goods(user_id, df)
print(f"\n{user_id}:")
print(purchased_goods)
