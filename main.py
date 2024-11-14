from Processing import DataAdd
from Collaborative_evaluate import main_evaluate
from ContentBased_Modeling import ListingRecommender
from ContentBased_evaluate import RecommenderEvaluator

def main():
    input_data_path = 'data/train_data(v0.1).csv'
    updated_data_path = 'data/train_updated_data_visitors_and_ratings.csv'
    test_data_path = 'data/test_data(v0.1).csv'
    updated_test_data_path = 'data/test_updated_data_visitors_and_ratings.csv'

    # print("1. Collaborative Filtering")

    # print("Train data processing...")
    # data_add_train = DataAdd(input_data_path, updated_data_path)
    # data_add_train.process_data()
    # print("Train data processing completed.")
    # print("=======================")

    # print("evaluation of train data...")
    # main_evaluate(updated_data_path, k=5)
    # print("evaluation completed.")
    # print("=======================")

    # print("Test data processing...")
    # data_add_test = DataAdd(test_data_path, updated_test_data_path)
    # data_add_test.process_data()
    # print("Test data processing completed.")
    # print("=======================")

    # print("evaluation on test data...")
    # main_evaluate(updated_test_data_path, k=5)
    # print("evaluation completed.")
    # print("=======================")

    print("\n2. Content-Based Filtering")
    
    # Train data evaluation
    print("Evaluation of train data...")
    recommender_train = ListingRecommender(updated_data_path)
    evaluator_train = RecommenderEvaluator(recommender_train)

    # 추천 및 평가 메트릭 계산
    listing_id = 35001175  # 예시 listing_id
    recommended_listings, precision, recall = evaluator_train.get_recommendations_with_eval(listing_id, topn=3)

        # 추천된 항목 및 메트릭 출력
    print("\nRecommended Listings:")
    print(recommended_listings)
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    print("Evaluation completed.")
    print("=======================")

    # Test data evaluation
    print("Evaluation of test data...")
    recommender_test = ListingRecommender(updated_test_data_path)
    evaluator_test = RecommenderEvaluator(recommender_test)
    
    # 추천 및 평가 메트릭 계산
    listing_id = 7699495  # 예시 listing_id
    recommended_listings, precision, recall = evaluator_test.get_recommendations_with_eval(listing_id, topn=3)

    # 추천된 항목 및 메트릭 출력
    print("\nRecommended Listings:")
    print(recommended_listings)
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    print("Evaluation completed.")
    print("=======================")

if __name__ == "__main__":
    main()