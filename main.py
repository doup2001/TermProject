from Collaborative_Processing import DataAdd
from Collaborative_evaluate import main_evaluate  # evaluate.py의 평가 함수 import

def main():
    # 경로 설정
    input_data_path = 'data/train_data(v0.1).csv'
    updated_data_path = 'data/train_updated_data_visitors_and_ratings.csv'
    test_data_path = 'data/test_data(v0.1).csv'
    updated_test_data_path = 'data/test_updated_data_visitors_and_ratings.csv'
    
    print("1. Collaborative Filtering")

    # 1. 학습 데이터 추가 및 저장 작업 수행
    print("Train data processing...")
    data_add_train = DataAdd(input_data_path, updated_data_path)
    data_add_train.process_data()
    print("Train data processing completed.")
    print("=======================")
    
    # 2. 학습 데이터 평가 작업 수행
    print("evaluation of train data...")
    main_evaluate(updated_data_path, k=5)  # 학습 데이터 평가 수행
    print("evaluation completed.")
    print("=======================")

    # 3. 테스트 데이터 추가 및 저장 작업 수행
    print("Test data processing...")
    data_add_test = DataAdd(test_data_path, updated_test_data_path)
    data_add_test.process_data()
    print("Test data processing completed.")
    print("=======================")

    # 4. 테스트 데이터 평가 작업 수행
    print("evaluation on test data...")
    main_evaluate(updated_test_data_path, k=5)  # 테스트 데이터 평가 수행
    print("evaluation completed.")
    print("=======================")

if __name__ == "__main__":
    main()
