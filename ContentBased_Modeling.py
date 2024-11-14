import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval

class ListingRecommender:
    def __init__(self, filepath):
        # 데이터 로드
        listing = pd.read_csv(filepath)
        listing['visitors'] = listing['visitors'].apply(literal_eval)
        
        # 필요한 열만 추출
        self.df_relevant = listing[['listing_id', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'price', 'city', 'visitors']]
        
        # 원-핫 인코딩 처리
        encoder = OneHotEncoder(sparse_output=False)
        encoded_features = encoder.fit_transform(self.df_relevant[['property_type', 'room_type']])
        
        # 원-핫 인코딩된 특성 데이터프레임에 추가
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['property_type', 'room_type']))
        self.encoded_df = pd.concat([self.df_relevant.reset_index(drop=True), encoded_df], axis=1)
        self.encoded_df = self.encoded_df.drop(columns=['property_type', 'room_type'])
        
        # price 형변환
        self.encoded_df['price'] = self.encoded_df['price'].replace({'\\$': ''}, regex=True).astype(float)
        
        # 필요한 특성만 선택
        self.features = self.encoded_df.drop(columns=['listing_id', 'city', 'visitors'])
        self.listing = listing

    def get_recommendations(self, listing_id, topn=10):
        idx = self.listing[self.listing['listing_id'] == listing_id].index[0]
        target_city = self.listing.loc[idx, 'city']
        
        # 같은 city에 속한 숙소 필터링
        listings_same_city = self.encoded_df[self.encoded_df['city'] == target_city].reset_index(drop=True)
        listings_original_same_city = self.df_relevant[self.df_relevant['city'] == target_city].reset_index(drop=True)
        
        # 대상 숙소 특징 벡터
        target_features = self.features.loc[idx].values.reshape(1, -1)
        city_features = listings_same_city.drop(columns=['listing_id', 'city', 'visitors']).values
        
        # 코사인 유사도 계산
        cosine_sim = cosine_similarity(target_features, city_features).flatten()
        sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)[1:topn+1]
        recommended_indices = [i[0] for i in sim_scores]
        
        # 추천 숙소 정보 반환
        recommended_listings = listings_original_same_city.iloc[recommended_indices]
        return recommended_listings[['listing_id', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'price', 'city']], recommended_listings['listing_id'].tolist()
