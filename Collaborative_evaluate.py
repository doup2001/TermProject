from Collaborative_Modeling import Modeling

def main_evaluate(data_path, k=5):
    # 데이터 경로로부터 모델링 인스턴스 생성
    modeling = Modeling(data_path)
    
    # 모델 평가 수행
    mean_precision_score, mae, rmse = modeling.evaluate_model(k=k)
    
    print(f"Evaluation completed.")
    print(f"Mean Precision@{k}: {mean_precision_score}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

if __name__ == "__main__":
    # 평가에 사용할 데이터 경로 설정
    data_path = 'data/updated_data_visitors_and_ratings.csv'
    main_evaluate(data_path, k=5)
