from Processing import DataAdd
from Collaborative_evaluate import main_evaluate
# from ContentBased_Modeling import ListingRecommender
from ContentBased_evaluate import RecommenderEvaluator

def main():
    input_data_path = 'data/train_data(v0.1).csv'
    updated_data_path = 'data/train_updated_data_visitors_and_ratings.csv'
    test_data_path = 'data/test_data(v0.1).csv'
    updated_test_data_path = 'data/test_updated_data_visitors_and_ratings.csv'

    print("1. Collaborative Filtering")

    print("Train data processing...")
    data_add_train = DataAdd(input_data_path, updated_data_path)
    data_add_train.process_data()
    print("Train data processing completed.")
    print("=======================")

    print("evaluation of train data...")
    main_evaluate(updated_data_path, k=5)
    print("evaluation completed.")
    print("=======================")

    print("Test data processing...")
    data_add_test = DataAdd(test_data_path, updated_test_data_path)
    data_add_test.process_data()
    print("Test data processing completed.")
    print("=======================")

    print("evaluation on test data...")
    main_evaluate(updated_test_data_path, k=5)
    print("evaluation completed.")
    print("=======================")

    print("\n2. Content-Based Filtering")
    
    print("evaluation of train data...")
    main_evaluate(updated_data_path, k=5)
    print("evaluation completed.")
    print("=======================")

    print("evaluation of test data...")
    evaluator = RecommenderEvaluator(updated_test_data_path)
    
     # 추천 및 평가 메트릭 계산
    listing_id = 7699495  # 예시 listing_id
    results = evaluator.evaluate(listing_id=listing_id, topn=10)

    # 추천된 항목 및 메트릭 출력
    print("\nRecommended Listings:")
    print(results['recommended_items'])
    print(f"\nPrecision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"NDCG: {results['ndcg']:.3f}")
    
    print("evaluation completed.")
    print("=======================")

if __name__ == "__main__":
    main()