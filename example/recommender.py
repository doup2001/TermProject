import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval

# 데이터 로드
listing = pd.read_csv('listing_with_visitors.csv')

# visitors 컬럼을 리스트로 변환 (문자열 형태의 리스트를 실제 리스트로 변환)
listing['visitors'] = listing['visitors'].apply(literal_eval)

# 필요한 열만 추출
df_relevant = listing[['listing_id', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'price', 'city', 'visitors']]

# 원-핫 인코딩 처리 (property_type, room_type)
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df_relevant[['property_type', 'room_type']])

# 원-핫 인코딩된 특성은 다시 데이터프레임에 추가
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['property_type', 'room_type']))
encoded_df = pd.concat([df_relevant.reset_index(drop=True), encoded_df], axis=1)
encoded_df = encoded_df.drop(columns=['property_type', 'room_type'])

# price 뒤에 $ 삭제 시키고 float 형변환
encoded_df['price'] = encoded_df['price'].replace({'\\$': ''}, regex=True).astype(float)

# listing_id와 city를 제외한 나머지 특성만 사용하여 유사도 계산 준비
features = encoded_df.drop(columns=['listing_id', 'city', 'visitors'])

def calculate_precision_recall(recommended_ids, target_visitor_list, all_visited_listings):
    """
    추천된 숙소들의 precision과 recall을 계산합니다.
    
    Parameters:
    recommended_ids (list): 추천된 숙소들의 listing_id 리스트
    target_visitor_list (list): 대상 숙소를 방문한 방문자 리스트
    all_visited_listings (dict): 각 방문자가 방문한 모든 숙소 리스트
    
    Returns:
    tuple: (precision, recall) 값
    """
    # 대상 숙소의 방문자들이 방문한 모든 숙소를 relevant items로 간주
    relevant_items = set()
    for visitor in target_visitor_list:
        relevant_items.update(all_visited_listings.get(visitor, []))
    
    # 추천된 아이템 중 relevant한 아이템 수 계산
    recommended_set = set(recommended_ids)
    relevant_and_recommended = relevant_items.intersection(recommended_set)
    
    # Precision과 Recall 계산
    precision = len(relevant_and_recommended) / len(recommended_set) if recommended_set else 0
    recall = len(relevant_and_recommended) / len(relevant_items) if relevant_items else 0
    
    return precision, recall

def get_recommendations_with_eval(listing_id, topn=10):
    """
    주어진 숙소에 대해 city가 같은 topn개의 유사한 숙소를 추천하고 평가 메트릭을 계산합니다.
    
    Parameters:
    listing_id (int): 대상 숙소의 listing_id
    topn (int): 상위 몇 개의 숙소를 추천할지 결정
    
    Returns:
    tuple: (추천 숙소 DataFrame, precision, recall)
    """
    # 대상 숙소의 인덱스 및 city 찾기
    idx = listing[listing['listing_id'] == listing_id].index[0]
    target_city = listing.loc[idx, 'city']
    
    # 같은 city에 속한 숙소만 필터링
    listings_same_city = encoded_df[encoded_df['city'] == target_city].reset_index(drop=True)
    listings_original_same_city = df_relevant[df_relevant['city'] == target_city].reset_index(drop=True)

    # 대상 숙소의 특징 벡터
    target_features = features.loc[idx].values.reshape(1, -1)
    
    # 같은 city에 있는 숙소들에 대한 특징 벡터
    city_features = listings_same_city.drop(columns=['listing_id', 'city', 'visitors']).values
    
    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(target_features, city_features).flatten()
    
    # 유사도 점수와 인덱스를 결합하여 정렬
    sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)

    # 자기 자신 제외한 상위 N개의 추천 숙소 선택
    sim_scores = sim_scores[1:topn+1]
    recommended_indices = [i[0] for i in sim_scores]
    
    # 추천 숙소 정보
    recommended_listings = listings_original_same_city.iloc[recommended_indices]
    
    # 평가 메트릭 계산을 위한 데이터 준비
    target_visitors = listing.loc[idx, 'visitors']
    recommended_ids = recommended_listings['listing_id'].tolist()
    
    # 각 방문자가 방문한 숙소 매핑 생성
    visitor_to_listings = {}
    for _, row in listing.iterrows():
        for visitor in row['visitors']:
            if visitor not in visitor_to_listings:
                visitor_to_listings[visitor] = []
            visitor_to_listings[visitor].append(row['listing_id'])
    
    # Precision과 Recall 계산
    precision, recall = calculate_precision_recall(
        recommended_ids,
        target_visitors,
        visitor_to_listings
    )
    
    return recommended_listings[['listing_id', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'price', 'city']], precision, recall

# 추천 및 평가 실행
recommendations, precision, recall = get_recommendations_with_eval(14130853, topn=10)
print("\nRecommended Listings:")
print(recommendations)
print(f"\nPrecision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# 전체 데이터셋에 대한 평균 성능 계산
def calculate_average_metrics(sample_size=100):
    """
    전체 데이터셋에서 무작위로 sample_size개의 숙소를 선택하여 평균 성능을 계산합니다.
    """
    all_precisions = []
    all_recalls = []
    
    # visitors가 있는 숙소만 선택
    valid_listings = listing[listing['visitors'].apply(len) > 0]['listing_id'].tolist()
    
    # sample_size개의 숙소를 무작위로 선택
    if len(valid_listings) > sample_size:
        sample_listings = np.random.choice(valid_listings, sample_size, replace=False)
    else:
        sample_listings = valid_listings
    
    for lid in sample_listings:
        try:
            _, precision, recall = get_recommendations_with_eval(lid, topn=10)
            all_precisions.append(precision)
            all_recalls.append(recall)
        except Exception as e:
            print(f"Error processing listing {lid}: {str(e)}")
            continue
    
    return {
        'mean_precision': np.mean(all_precisions),
        'mean_recall': np.mean(all_recalls),
        'std_precision': np.std(all_precisions),
        'std_recall': np.std(all_recalls)
    }

# 평균 성능 계산
print("\nCalculating average metrics...")
metrics = calculate_average_metrics(sample_size=100)
print("\nAverage Performance Metrics:")
print(f"Mean Precision: {metrics['mean_precision']:.3f} (±{metrics['std_precision']:.3f})")
print(f"Mean Recall: {metrics['mean_recall']:.3f} (±{metrics['std_recall']:.3f})")