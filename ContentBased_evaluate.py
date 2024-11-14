import numpy as np
import pandas as pd
from ContentBased_Modeling import ListingRecommender

class RecommenderEvaluator:
    def __init__(self, listing_recommender):
        self.listing_recommender = listing_recommender
        self.listing = listing_recommender.listing
    
    def calculate_precision_recall(self, recommended_ids, target_visitor_list, all_visited_listings):
        relevant_items = set()
        for visitor in target_visitor_list:
            relevant_items.update(all_visited_listings.get(visitor, []))
        
        recommended_set = set(recommended_ids)
        relevant_and_recommended = relevant_items.intersection(recommended_set)
        
        precision = len(relevant_and_recommended) / len(recommended_set) if recommended_set else 0
        recall = len(relevant_and_recommended) / len(relevant_items) if relevant_items else 0
        return precision, recall

    def get_recommendations_with_eval(self, listing_id, topn=10):
        recommended_listings, recommended_ids = self.listing_recommender.get_recommendations(listing_id, topn=topn)
        
        idx = self.listing[self.listing['listing_id'] == listing_id].index[0]
        target_visitors = self.listing.loc[idx, 'visitors']
        
        visitor_to_listings = {}
        for _, row in self.listing.iterrows():
            for visitor in row['visitors']:
                if visitor not in visitor_to_listings:
                    visitor_to_listings[visitor] = []
                visitor_to_listings[visitor].append(row['listing_id'])
        
        precision, recall = self.calculate_precision_recall(
            recommended_ids,
            target_visitors,
            visitor_to_listings
        )
        
        return recommended_listings, precision, recall