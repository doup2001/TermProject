from Processing import DataAdd
from Collaborative_Modeling import Modeling
from Collaborative_evaluate import ModelEvaluator
from ContentBased_Modeling import ListingRecommender
from ContentBased_evaluate import RecommenderEvaluator

def main():
    input_data_path = 'data/train_data(v0.1).csv'
    updated_data_path = 'data/train_updated_data_visitors_and_ratings.csv'
    test_data_path = 'data/test_data(v0.1).csv'
    updated_test_data_path = 'data/test_updated_data_visitors_and_ratings.csv'
    listing = 'data/listing_with_visitors.csv'

    print("Train data processing...")
    data_add_train = DataAdd(input_data_path, updated_data_path)
    data_add_train.process_data()
    print("Train data processing completed.")
    print("=======================")

    print("Test data processing...")
    data_add_test = DataAdd(test_data_path, updated_test_data_path)
    data_add_test.process_data()
    print("Test data processing completed.")
    print("=======================")

    print("1. Collaborative Filtering")    

    print("evaluation of train data...")
    modeling = Modeling(updated_data_path)
    evaluator = ModelEvaluator(modeling, k=5)
    mean_precision_score, mae, rmse = evaluator.evaluate()
    evaluator.print_results(mean_precision_score, mae, rmse)
    print("evaluation completed.")
    print("=======================")

    print("evaluation on test data...")
    modeling = Modeling(updated_test_data_path)
    evaluator = ModelEvaluator(modeling, k=5)
    mean_precision_score, mae, rmse = evaluator.evaluate()
    evaluator.print_results(mean_precision_score, mae, rmse)
    print("evaluation completed.")
    print("=======================")

    print("\n2. Content-Based Filtering")

        # ListingRecommender 객체를 생성 (파일 경로를 실제로 지정)
    recommender_train = ListingRecommender(listing)

    # RecommenderEvaluator 객체를 생성
    evaluator = RecommenderEvaluator(recommender_train)

    # 유저 ID를 입력받아서 해당 유저의 추천 목록을 계산
    user_id = 5127407  # 실제 유저 ID 입력
    recommended_listings, precision_at_k_score = evaluator.get_recommendations_with_eval(user_id, topn=11)

    print(f"\nPrecision@10: {precision_at_k_score:.3f}")

if __name__ == "__main__":
    main()
