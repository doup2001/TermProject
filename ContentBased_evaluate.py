import numpy as np
import pandas as pd
from ast import literal_eval
from ContentBased_Modeling import ListingRecommender

class RecommenderEvaluator:
    def __init__(self, listing_recommender):
        self.listing_recommender = listing_recommender
        self.listing = listing_recommender.listing

    def precision_at_k(self, recommended_ids, relevant_items, k):
        recommended_at_k = recommended_ids[:k]  # 상위 k개 추천된 항목
        hits = 0
        
        # 추천된 항목 ID와 실제 방문한 항목 ID를 비교
        for rec_id in recommended_at_k:
            if rec_id in relevant_items:
                hits += 1  # 실제 방문한 항목이 추천된 항목에 포함되었으면 hit
        
        precision_at_k_value = hits / k  # Precision@k 계산
        return precision_at_k_value

    def get_user_visited_listings(self, user_id):
        """주어진 user_id로 방문한 숙소 listing_id 목록을 추출"""
        def safe_eval(x):
            # 이미 x가 리스트 형태일 경우
            return x  # 리스트 그대로 반환

        # 유저 방문 정보를 안전하게 평가
        user_visits = self.listing[self.listing['visitors'].apply(lambda x: user_id in safe_eval(x))]
        return user_visits['listing_id'].tolist()

    def get_recommendations_with_eval(self, user_id, topn=10):
        # 유저가 방문한 listing_id 목록 추출
        user_visited_listings = self.get_user_visited_listings(user_id)
        
        # 유저가 방문한 listing_id로 추천 생성
        recommended_listings, recommended_ids = self.listing_recommender.get_recommendations_with_user_preference(user_visited_listings, topn=topn)
        
        print(f"User ID: {user_id}")
        # print(f"Recommended IDs: {recommended_ids}")
        # print(f"Target Visitors (Visited Listings): {user_visited_listings}")
        
        # Precision at k 계산
        precision_at_k_score = self.precision_at_k(recommended_ids, user_visited_listings, k=topn)
        
        # 추천된 listing 정보 출력
        print("\nRecommended Listings Details:")
        for listing_id in recommended_ids[:topn]:
            listing_info = self.listing[self.listing['listing_id'] == listing_id]
            if not listing_info.empty:
                print(f"\nListing ID: {listing_id}")
                print(listing_info[['listing_id', 'property_type', 'room_type', 'accommodates','price', 'bedrooms','city']])  # 필요한 정보만 출력
            else:
                print(f"Listing ID: {listing_id} not found in the dataset.")
        
        return recommended_listings, precision_at_k_score