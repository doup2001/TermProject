from DataAdd import DataAdd
from Modeling import Modeling

def main():
    # 경로 설정
    input_data_path = 'data/train_data(v0.1).csv'
    updated_data_path = 'data/train_updated_data_visitors_and_ratings.csv'
    test_data_path = 'data/test_data(v0.1).csv'  # 테스트 데이터 경로 추가
    updated_test_data_path = 'data/test_updated_data_visitors_and_ratings.csv'
    
    print("Collaborative Filtering")
    print("=======================.\n")

    # 1. 데이터 추가 및 저장 작업 수행
    print("Starting train data processing...")
    data_add = DataAdd(input_data_path, updated_data_path)
    data_add.process_data()
    print("Data processing completed.")
    print("=======================.\n")
    
    # 2. 모델 평가 작업 수행 (학습 데이터)
    print("Starting modeling on training data...")
    modeling_train = Modeling(updated_data_path)
    mean_precision_score_train = modeling_train.evaluate_model(k=5)
    print("Training data evaluation completed.")
    print(f"Mean Precision@5 score on training data: {mean_precision_score_train}")
    print("=======================.\n")
    
    # 3. 데이터 추가 및 저장 작업 수행
    print("Starting test data processing...")
    data_add = DataAdd(test_data_path, updated_test_data_path)
    data_add.process_data()
    print("Data processing completed.")
    print("=======================.\n")

    # 4. 모델 평가 작업 수행 (테스트 데이터)
    print("Starting modeling on test data...")
    modeling_test = Modeling(updated_test_data_path)
    mean_precision_score_test = modeling_test.evaluate_model(k=5)
    print("Test data evaluation completed.")
    print(f"Mean Precision@5 score on test data: {mean_precision_score_test}")
    print("=======================.\n")

if __name__ == "__main__":
    main()
