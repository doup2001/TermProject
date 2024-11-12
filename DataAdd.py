import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 데이터셋 업데이트 - 희소 사용자-아이템 데이터 및 가상 평점 추가
# Load original data
data = pd.read_csv('C:/Users/82109/Desktop/data/train_data(v0.1).csv')

# Define a visitor pool to assign sparse visitors
visitor_pool = [f"visitor_{i}" for i in range(1, 101)]  # 100명의 가상 방문객 생성

# Assign sparse visitors to each listing
def assign_sparse_visitors(row, visitor_pool, num_visitors=4):
    return np.random.choice(visitor_pool, num_visitors, replace=False).tolist()

data['visitors'] = data.apply(lambda row: assign_sparse_visitors(row, visitor_pool), axis=1)

# Assign varied user ratings with a range of -5 to +5 around the base review score, with limits between 0 and 100
def assign_varied_user_rating_with_limits(row, variation_range=(-5, 5)):
    base_score = row['review_scores_rating']
    return [max(0, min(100, int(base_score + np.random.randint(variation_range[0], variation_range[1]))))
            for _ in range(len(row['visitors']))]

data['user_ratings'] = data.apply(assign_varied_user_rating_with_limits, axis=1)

# Save the updated data
updated_file_path = 'C:/Users/82109/Desktop/data/updated_data_visitors_and_ratings.csv'
data.to_csv(updated_file_path, index=False)