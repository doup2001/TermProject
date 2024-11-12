import pandas as pd
import numpy as np

def update_dataset_with_visitors_and_ratings(file_path, visitor_count=100, num_visitors_per_listing=4, variation_range=(-5, 5)):
    # 데이터 불러오기
    data = pd.read_csv(file_path)
    
    # 가상 방문자 생성
    visitor_pool = [f"visitor_{i}" for i in range(1, visitor_count + 1)]
    
    # 각 리스트에 희소 방문자 할당
    def assign_sparse_visitors(row):
        return np.random.choice(visitor_pool, num_visitors_per_listing, replace=False).tolist()

    data['visitors'] = data.apply(assign_sparse_visitors, axis=1)
    
    # 사용자 평점 변동 범위 설정
    def assign_varied_user_rating_with_limits(row):
        base_score = row['review_scores_rating']
        return [max(0, min(100, int(base_score + np.random.randint(variation_range[0], variation_range[1]))))
                for _ in range(len(row['visitors']))]
    
    data['user_ratings'] = data.apply(assign_varied_user_rating_with_limits, axis=1)
    
    return data
