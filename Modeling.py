import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 업데이트된 데이터 불러오기 및 사용자-아이템 매트릭스 생성
user_item_data = pd.read_csv('C:/Users/82109/Desktop/data/updated_data_visitors_and_ratings.csv')

# Convert columns from string to list for 'visitors' and 'user_ratings' then explode them
user_item_data['visitors'] = user_item_data['visitors'].apply(eval)
user_item_data['user_ratings'] = user_item_data['user_ratings'].apply(eval)
user_item_data_exploded = user_item_data.explode(['visitors', 'user_ratings']).rename(
    columns={'visitors': 'user_id', 'user_ratings': 'user_rating'}
)

# 샘플링 - 전체 데이터의 20%만 사용
sampled_data = user_item_data_exploded.sample(frac=0.2, random_state=42)

# 유사도 계산을 위한 아이템-사용자 매트릭스 생성
sampled_data['interaction_str'] = sampled_data.apply(
    lambda x: f"{x['user_id']}_{x['user_rating']}", axis=1
)
listing_interactions = sampled_data.groupby('listing_id')['interaction_str'].apply(lambda x: ' '.join(x))

# Create a user-item matrix with CountVectorizer
count_vectorizer = CountVectorizer()
interaction_matrix = count_vectorizer.fit_transform(listing_interactions)

# Calculate cosine similarity between listings
listing_similarity = cosine_similarity(interaction_matrix, interaction_matrix)
listing_indices = listing_interactions.index.tolist()

# 협업 필터링 추천 함수
def get_similar_listings(listing_id, similarity_matrix=listing_similarity, top_k=5):
    if listing_id not in listing_indices:
        return "Listing ID not found."
    
    idx = listing_indices.index(listing_id)
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k + 1]  # 가장 유사한 숙소 top_k개 반환

    similar_listing_indices = [listing_indices[i[0]] for i in sim_scores]
    return sampled_data[sampled_data['listing_id'].isin(similar_listing_indices)][['listing_id', 'user_id']]

# Precision@K 평가 함수
def user_precision_at_k(user_id, similarity_matrix=listing_similarity, k=5):
    user_high_rated_listings = sampled_data[
        (sampled_data['user_id'] == user_id) & (sampled_data['user_rating'] >= 90)
    ]['listing_id'].unique()

    if len(user_high_rated_listings) == 0:
        return 0  # No high ratings for this user
    
    sample_listing_id = user_high_rated_listings[0]
    recommended_listings = get_similar_listings(sample_listing_id, similarity_matrix, top_k=k)
    recommended_ids = recommended_listings['listing_id'].tolist()
    relevant_at_k = set(recommended_ids) & set(user_high_rated_listings)
    precision_score = len(relevant_at_k) / k
    return precision_score

# 전체 사용자에 대한 Mean Precision@K 계산
def mean_precision_at_k_all_users(user_data, similarity_matrix, k=5):
    unique_users = user_data['user_id'].unique()
    precision_scores = []
    for user_id in unique_users:
        precision = user_precision_at_k(user_id, similarity_matrix, k=k)
        if isinstance(precision, float):
            precision_scores.append(precision)
    
    mean_precision = np.mean(precision_scores) if precision_scores else 0
    return mean_precision

# Calculate Mean Precision@5 across all users
mean_precision_score = mean_precision_at_k_all_users(sampled_data, listing_similarity, k=5)

# Print out the final Mean Precision@5 score as the overall performance metric
print("Mean Precision@5 for all users:", mean_precision_score)