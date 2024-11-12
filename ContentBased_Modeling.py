import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from ast import literal_eval

class ListingRecommender:
    def __init__(self, data_path):
        """
        data_path: 데이터 파일의 경로
        """
        self.df = pd.read_csv(data_path)
        self.similarity_df = None

    def calculate_user_score(self):
        """
        'visitors'와 'user_ratings'를 바탕으로 각 숙소에 대한 유저 점수를 계산하여 'user_score' 열을 추가합니다.
        """
        self.df['user_score'] = self.df['user_ratings'].apply(lambda x: np.mean(literal_eval(x)))

    def preprocess_features(self):
        """
        추천 시스템에 사용할 Feature들을 선택하고 전처리하여 feature_df를 생성합니다.
        범주형 변수는 원-핫 인코딩, 수치형 변수는 표준화를 수행합니다.
        """
        features_df = self.df[['property_type', 'room_type', 'accommodates', 'bedrooms', 'price']].copy()

        # 'price' 열에서 '$' 기호를 제거하고 float로 변환합니다
        features_df['price'] = features_df['price'].replace('[\$,]', '', regex=True).astype(float)

        # 범주형 변수에 대해 원-핫 인코딩 수행
        features_df = pd.get_dummies(features_df, columns=['property_type', 'room_type'])

        # 수치형 변수에 대해 표준화 수행
        scaler = StandardScaler()
        features_df[['accommodates', 'bedrooms', 'price']] = scaler.fit_transform(features_df[['accommodates', 'bedrooms', 'price']])
        
        return features_df

    def calculate_similarity(self):
        """
        전처리된 Feature들을 기반으로 코사인 유사도를 계산하여 similarity_df를 생성합니다.
        """
        features_df = self.preprocess_features()
        similarity_matrix = cosine_similarity(features_df)
        
        # similarity_df의 인덱스와 열을 listing_id로 설정
        if 'listing_id' not in self.df.columns:
            raise ValueError("The dataset does not contain 'listing_id' column.")

        self.similarity_df = pd.DataFrame(similarity_matrix, index=self.df['listing_id'], columns=self.df['listing_id'])

    def get_recommendations(self, listing_id, top_n=10):
        """
        특정 숙소와 유사한 상위 top_n개의 숙소를 추천합니다.
        """
        if self.similarity_df is None:
            self.calculate_similarity()

        # listing_id가 similarity_df에 존재하는지 확인
        if listing_id not in self.similarity_df.index:
            raise ValueError(f"Listing ID {listing_id} not found in similarity dataframe.")

        similar_listings = self.similarity_df[listing_id].sort_values(ascending=False)
        recommendations = similar_listings.iloc[1:top_n+1].index.tolist()
        return recommendations
