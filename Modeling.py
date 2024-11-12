import pandas as pd
import numpy as np
import ast

def calculate_mean_precision(df, k=5, sample_frac=0.2, random_state=42):
    # 'visitors'와 'user_ratings' 열이 리스트로 되어 있는지 확인하고, 아니면 변환
    df['visitors'] = df['visitors'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['user_ratings'] = df['user_ratings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    df_exploded = df.explode(['visitors', 'user_ratings']).rename(
        columns={'visitors': 'user_id', 'user_ratings': 'user_rating'}
    )

    # 샘플링 - 전체 데이터의 일정 비율만 사용
    sampled_data = df_exploded.sample(frac=sample_frac, random_state=random_state)

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
    def get_similar_listings(listing_id, similarity_matrix=listing_similarity, top_k=k):
        if listing_id not in listing_indices:
            return "Listing ID not found."
        
        idx = listing_indices.index(listing_id)
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k + 1]  # 가장 유사한 숙소 top_k개 반환

        similar_listing_indices = [listing_indices[i[0]] for i in sim_scores]
        return sampled_data[sampled_data['listing_id'].isin(similar_listing_indices)][['listing_id', 'user_id']]

    # Precision@K 평가 함수
    def user_precision_at_k(user_id, similarity_matrix=listing_similarity, k=k):
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
    unique_users = sampled_data['user_id'].unique()
    precision_scores = [user_precision_at_k(user_id, listing_similarity, k) for user_id in unique_users]
    mean_precision = np.mean(precision_scores) if precision_scores else 0

    return mean_precision
