import numpy as np
import pandas as pd
from ContentBased_Modeling import ListingRecommender


class RecommenderEvaluator:
    def __init__(self, data_path):
        self.recommender = ListingRecommender(data_path)
        self.recommender.calculate_user_score()  # 추천 시스템 초기화 및 계산

    @staticmethod
    def calculate_precision(recommended, relevant):
        # Precision: 추천 항목 중 실제로 관련 있는 항목의 비율
        return len(set(recommended) & set(relevant)) / len(recommended) if len(recommended) > 0 else 0

    @staticmethod
    def calculate_recall(recommended, relevant):
        # Recall: 실제 관련 있는 항목 중 추천된 항목의 비율
        return len(set(recommended) & set(relevant)) / len(relevant) if len(relevant) > 0 else 0

    @staticmethod
    def calculate_ndcg(recommended, relevant):
        # NDCG: 추천 항목이 실제 관련 항목과 얼마나 잘 일치하는지 계산
        dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(recommended) if item in relevant])
        idcg = sum([1 / np.log2(i + 2) for i in range(len(relevant))])
        return dcg / idcg if idcg > 0 else 0

    def evaluate(self, listing_id, topn=10):
        """
        추천된 항목들과 실제 관련 항목들에 대해 평가 메트릭을 계산합니다.
        ListingRecommender를 내부적으로 사용하여 추천과 평가를 진행
        """
        # 추천된 항목들
        recommendations = self.recommender.get_recommendations(listing_id, top_n=topn)
        
        # 실제 관련 항목들 (이 부분을 사용자 피드백이나 비슷한 숙소들을 기반으로 정의)
        relevant_items = self.recommender.get_recommendations(listing_id, top_n=topn * 2)[:topn]
        
        # 평가 메트릭 계산
        precision = self.calculate_precision(recommendations, relevant_items)
        recall = self.calculate_recall(recommendations, relevant_items)
        ndcg = self.calculate_ndcg(recommendations, relevant_items)

        # 추천된 항목들과 메트릭들을 함께 반환
        return {
            'recommended_items': recommendations,
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg
        }


def main_evaluate(data_path, topn=10):
    # 데이터 로드
    data = pd.read_csv(data_path)
    
    # listing_id 자동 선택 (여기선 첫 번째 항목을 사용)
    listing_id = data['listing_id'].iloc[0]  # 첫 번째 listing_id를 자동으로 선택
    # 또는 무작위로 하나를 선택하려면 다음을 사용
    # listing_id = data['listing_id'].sample(n=1).iloc[0]
    
    # 데이터 경로로부터 평가 인스턴스 생성
    evaluator = RecommenderEvaluator(data_path)
    
    # 모델 평가 수행
    precision, recall, ndcg = evaluator.evaluate(listing_id, topn=topn)
    
    print(f"Evaluation completed.")
    print(f"Precision@{topn}: {precision}")
    print(f"Recall@{topn}: {recall}")
    print(f"NDCG@{topn}: {ndcg}")


if __name__ == "__main__":
    # 평가에 사용할 데이터 경로 설정
    data_path = 'data/updated_data_visitors_and_ratings.csv'
    main_evaluate(data_path, topn=10)
