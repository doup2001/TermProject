import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Modeling:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.data['visitors'] = self.data['visitors'].apply(eval)
        self.data['user_ratings'] = self.data['user_ratings'].apply(eval)
        self.data = self.data.explode(['visitors', 'user_ratings']).rename(
            columns={'visitors': 'user_id', 'user_ratings': 'user_rating'}
        )
        self.sampled_data = self.data.sample(frac=0.2, random_state=42)
        self.listing_similarity = None
        self.listing_indices = []

    def create_similarity_matrix(self):
        self.sampled_data['interaction_str'] = self.sampled_data.apply(
            lambda x: f"{x['user_id']}_{x['user_rating']}", axis=1
        )
        listing_interactions = self.sampled_data.groupby('listing_id')['interaction_str'].apply(lambda x: ' '.join(x))
        count_vectorizer = CountVectorizer()
        interaction_matrix = count_vectorizer.fit_transform(listing_interactions)
        self.listing_similarity = cosine_similarity(interaction_matrix, interaction_matrix)
        self.listing_indices = listing_interactions.index.tolist()

    def get_similar_listings(self, listing_id, top_k=5):
        if listing_id not in self.listing_indices:
            return "Listing ID not found."
        idx = self.listing_indices.index(listing_id)
        sim_scores = sorted(enumerate(self.listing_similarity[idx]), key=lambda x: x[1], reverse=True)[1:top_k + 1]
        similar_listing_indices = [self.listing_indices[i[0]] for i in sim_scores]
        return self.sampled_data[self.sampled_data['listing_id'].isin(similar_listing_indices)][['listing_id', 'user_id', 'user_rating']]

    def user_precision_at_k(self, user_id, k=5):
        user_high_rated_listings = self.sampled_data[
            (self.sampled_data['user_id'] == user_id) & (self.sampled_data['user_rating'] >= 90)
        ]['listing_id'].unique()
        if len(user_high_rated_listings) == 0:
            return 0  # No high ratings for this user
        sample_listing_id = user_high_rated_listings[0]
        recommended_listings = self.get_similar_listings(sample_listing_id, top_k=k)
        recommended_ids = recommended_listings['listing_id'].tolist()
        relevant_at_k = set(recommended_ids) & set(user_high_rated_listings)
        return len(relevant_at_k) / k

    def mean_precision_at_k_all_users(self, k=5):
        unique_users = self.sampled_data['user_id'].unique()
        precision_scores = [
            self.user_precision_at_k(user_id, k=k) for user_id in unique_users
            if isinstance(self.user_precision_at_k(user_id, k=k), float)
        ]
        return np.mean(precision_scores) if precision_scores else 0

    def calculate_mae_rmse(self):
        """Calculate MAE and RMSE for user ratings in the sampled data."""
        actual_ratings = self.sampled_data['user_rating'].values
        predicted_ratings = np.random.randint(1, 101, len(actual_ratings))  # Placeholder for predicted ratings
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        return mae, rmse

    def evaluate_model(self, k=5):
        self.create_similarity_matrix()
        mean_precision_score = self.mean_precision_at_k_all_users(k=k)
        mae, rmse = self.calculate_mae_rmse()
        return mean_precision_score, mae, rmse
