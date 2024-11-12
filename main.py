from DataAdd import update_dataset_with_visitors_and_ratings
from Modeling import calculate_mean_precision

def main():
    # 테스트할 CSV 파일 경로
    file_path = 'data/test_data(v0.1).csv'
    
    # 1. 데이터 업데이트
    updated_data = update_dataset_with_visitors_and_ratings(file_path, visitor_count=100, num_visitors_per_listing=4)
    
    # 2. 추천 시스템으로 Precision@5 계산
    mean_precision_score = calculate_mean_precision(updated_data, k=5)
    
    # 3. 결과 출력
    print("Mean Precision@5 for all users:", mean_precision_score)

if __name__ == "__main__":
    main()
