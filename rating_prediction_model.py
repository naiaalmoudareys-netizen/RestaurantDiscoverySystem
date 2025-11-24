"""
Task 2.1: Feature Engineering & Model Implementation
Restaurant Rating Prediction Model

Features:
- Structured data: cuisine, price, location
- Unstructured: review text (sentiment, length, keywords)
- Time-series: dining trends (popularity, seasonality)
- User features: age, dining frequency, preferences

Model: XGBoost with hyperparameter tuning
Evaluation: RMSE, MAE, R²
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# For model interpretability
import shap


class RatingPredictionModel:
    """
    Restaurant Rating Prediction Model
    
    Combines structured, unstructured, and time-series features to predict
    restaurant ratings.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.feature_names = []
        self.is_trained = False
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data sources."""
        print("Loading data sources...")
        
        # Load restaurants
        with open('restaurant.json', 'r', encoding='utf-8') as f:
            restaurants_data = json.load(f)
        restaurants_df = pd.DataFrame(restaurants_data)
        print(f"  - Loaded {len(restaurants_df)} restaurants")
        
        # Load reviews
        reviews_df = pd.read_csv('reviews.csv')
        print(f"  - Loaded {len(reviews_df)} reviews")
        
        # Load user data
        user_df = pd.read_csv('user_data.csv')
        print(f"  - Loaded {len(user_df)} users")
        
        # Load dining trends
        trends_df = pd.read_csv('dining_trends.csv')
        trends_df['date'] = pd.to_datetime(trends_df['date'])
        print(f"  - Loaded {len(trends_df)} trend records")
        
        return restaurants_df, reviews_df, user_df, trends_df
    
    def engineer_structured_features(self, restaurants_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from structured restaurant data."""
        print("\nEngineering structured features...")
        
        features = pd.DataFrame()
        features['restaurant_id'] = restaurants_df['id']
        
        # Cuisine encoding
        if 'cuisine' not in self.label_encoders:
            self.label_encoders['cuisine'] = LabelEncoder()
            features['cuisine_encoded'] = self.label_encoders['cuisine'].fit_transform(restaurants_df['cuisine'])
        else:
            features['cuisine_encoded'] = self.label_encoders['cuisine'].transform(restaurants_df['cuisine'])
        
        # Location encoding
        if 'location' not in self.label_encoders:
            self.label_encoders['location'] = LabelEncoder()
            features['location_encoded'] = self.label_encoders['location'].fit_transform(restaurants_df['location'])
        else:
            features['location_encoded'] = self.label_encoders['location'].transform(restaurants_df['location'])
        
        # Price range features
        price_ranges = restaurants_df['price_range'].str.extract(r'AED\s*(\d+)\s*-\s*(\d+)')
        features['price_min'] = pd.to_numeric(price_ranges[0], errors='coerce').fillna(0)
        features['price_max'] = pd.to_numeric(price_ranges[1], errors='coerce').fillna(0)
        features['price_avg'] = (features['price_min'] + features['price_max']) / 2
        
        # Handle price ranges with "+" (e.g., "AED 200 - 300+")
        plus_ranges = restaurants_df['price_range'].str.contains('\+', na=False)
        features.loc[plus_ranges, 'price_max'] = features.loc[plus_ranges, 'price_max'] * 1.5  # Estimate upper bound
        features.loc[plus_ranges, 'price_avg'] = (features.loc[plus_ranges, 'price_min'] + features.loc[plus_ranges, 'price_max']) / 2
        
        # Amenities count
        features['amenities_count'] = restaurants_df['amenities'].str.count(',') + 1
        features['amenities_count'] = features['amenities_count'].fillna(0)
        
        # Has outdoor seating
        features['has_outdoor_seating'] = restaurants_df['amenities'].str.contains('Outdoor Seating', case=False, na=False).astype(int)
        
        # Has live music
        features['has_live_music'] = restaurants_df['amenities'].str.contains('Live Music', case=False, na=False).astype(int)
        
        # Attributes count
        features['attributes_count'] = restaurants_df['attributes'].str.count(',') + 1
        features['attributes_count'] = features['attributes_count'].fillna(0)
        
        # Is romantic
        features['is_romantic'] = restaurants_df['attributes'].str.contains('Romantic', case=False, na=False).astype(int)
        
        # Is fine dining
        features['is_fine_dining'] = restaurants_df['attributes'].str.contains('Fine Dining', case=False, na=False).astype(int)
        
        print(f"  - Created {len(features.columns) - 1} structured features")
        
        return features
    
    def engineer_text_features(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from unstructured review text."""
        print("\nEngineering text features...")
        
        text_features = pd.DataFrame()
        text_features['review_id'] = reviews_df['review_id']
        text_features['restaurant_id'] = reviews_df['restaurant_id']
        
        # Review length
        text_features['review_length'] = reviews_df['review_text'].str.len()
        
        # Word count
        text_features['word_count'] = reviews_df['review_text'].str.split().str.len()
        
        # Sentiment indicators (simple keyword-based)
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'delicious', 'outstanding']
        negative_words = ['terrible', 'awful', 'bad', 'poor', 'disappointing', 'horrible', 'worst', 'waste']
        
        text_features['positive_word_count'] = reviews_df['review_text'].str.lower().str.count('|'.join(positive_words))
        text_features['negative_word_count'] = reviews_df['review_text'].str.lower().str.count('|'.join(negative_words))
        text_features['sentiment_score'] = text_features['positive_word_count'] - text_features['negative_word_count']
        
        # Exclamation marks (enthusiasm indicator)
        text_features['exclamation_count'] = reviews_df['review_text'].str.count('!')
        
        # Question marks (uncertainty indicator)
        text_features['question_count'] = reviews_df['review_text'].str.count('\?')
        
        # Has specific keywords
        text_features['mentions_service'] = reviews_df['review_text'].str.lower().str.contains('service|server|staff|waiter', case=False, na=False).astype(int)
        text_features['mentions_food'] = reviews_df['review_text'].str.lower().str.contains('food|dish|meal|cuisine|taste', case=False, na=False).astype(int)
        text_features['mentions_ambiance'] = reviews_df['review_text'].str.lower().str.contains('ambiance|atmosphere|environment|decor', case=False, na=False).astype(int)
        text_features['mentions_price'] = reviews_df['review_text'].str.lower().str.contains('price|cost|expensive|cheap|value', case=False, na=False).astype(int)
        
        # Aggregate by restaurant
        restaurant_text_features = text_features.groupby('restaurant_id').agg({
            'review_length': ['mean', 'std'],
            'word_count': ['mean', 'std'],
            'sentiment_score': ['mean', 'sum'],
            'positive_word_count': 'mean',
            'negative_word_count': 'mean',
            'exclamation_count': 'mean',
            'question_count': 'mean',
            'mentions_service': 'sum',
            'mentions_food': 'sum',
            'mentions_ambiance': 'sum',
            'mentions_price': 'sum'
        }).reset_index()
        
        # Flatten column names
        restaurant_text_features.columns = ['restaurant_id'] + ['_'.join(col).strip() for col in restaurant_text_features.columns[1:]]
        
        print(f"  - Created {len(restaurant_text_features.columns) - 1} text features per restaurant")
        
        return restaurant_text_features
    
    def engineer_timeseries_features(self, trends_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from time-series dining trends."""
        print("\nEngineering time-series features...")
        
        # Merge reviews with trends by date and cuisine
        reviews_df['date'] = pd.to_datetime(reviews_df['date'])
        
        # Get cuisine for each restaurant from reviews (merge with restaurant data)
        # For now, aggregate trends by cuisine type
        trends_agg = trends_df.groupby('cuisine_type').agg({
            'popularity_score': ['mean', 'std', 'max'],
            'avg_price': 'mean',
            'booking_lead_time_days': 'mean'
        }).reset_index()
        
        trends_agg.columns = ['cuisine_type', 'trend_popularity_mean', 'trend_popularity_std', 'trend_popularity_max', 
                             'trend_avg_price', 'trend_booking_lead_time']
        
        # Seasonality features
        trends_df['month'] = trends_df['date'].dt.month
        trends_df['is_peak_season'] = trends_df['season'].str.contains('Peak', case=False, na=False).astype(int)
        trends_df['is_weekend'] = (trends_df['day_type'] == 'Weekend').astype(int)
        trends_df['is_holiday'] = trends_df['is_holiday'].astype(int)
        
        # Encode weather impact
        weather_map = {'Favorable Weather': 2, 'Neutral': 1, 'Unfavorable Weather': 0}
        trends_df['weather_impact_encoded'] = trends_df['weather_impact_category'].map(weather_map).fillna(1)
        
        seasonality = trends_df.groupby('cuisine_type').agg({
            'is_peak_season': 'mean',
            'is_weekend': 'mean',
            'is_holiday': 'mean',
            'weather_impact_encoded': 'mean'
        }).reset_index()
        seasonality.columns = ['cuisine_type', 'peak_season_ratio', 'weekend_ratio', 'holiday_ratio', 'weather_impact']
        
        # Merge trend features
        trends_features = trends_agg.merge(seasonality, on='cuisine_type', how='left')
        
        print(f"  - Created {len(trends_features.columns) - 1} trend features per cuisine")
        
        return trends_features
    
    def engineer_user_features(self, user_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from user data."""
        print("\nEngineering user features...")
        
        # Merge user data with reviews
        user_reviews = reviews_df.merge(user_df, on='user_id', how='left')
        
        # Aggregate user features by restaurant
        restaurant_user_features = user_reviews.groupby('restaurant_id').agg({
            'age': 'mean',
            'dining_frequency': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Weekly',  # Most common
            'avg_rating_given': 'mean',
            'total_reviews_written': 'mean'
        }).reset_index()
        
        # Encode dining frequency
        frequency_map = {'Daily': 7, 'Bi-Weekly': 3.5, 'Weekly': 1, 'Monthly': 0.25}
        restaurant_user_features['dining_frequency_encoded'] = restaurant_user_features['dining_frequency'].map(frequency_map).fillna(1)
        
        # Drop original dining_frequency
        restaurant_user_features = restaurant_user_features.drop('dining_frequency', axis=1)
        
        print(f"  - Created {len(restaurant_user_features.columns) - 1} user features per restaurant")
        
        return restaurant_user_features
    
    def create_feature_matrix(self, restaurants_df: pd.DataFrame, reviews_df: pd.DataFrame, 
                             user_df: pd.DataFrame, trends_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create complete feature matrix and target variable."""
        print("\nCreating feature matrix...")
        
        # Start with structured features
        features = self.engineer_structured_features(restaurants_df)
        
        # Add text features
        text_features = self.engineer_text_features(reviews_df)
        features = features.merge(text_features, on='restaurant_id', how='left')
        
        # Add trend features (merge by cuisine)
        trends_features = self.engineer_timeseries_features(trends_df, reviews_df)
        features = features.merge(restaurants_df[['id', 'cuisine']], left_on='restaurant_id', right_on='id', how='left')
        features = features.merge(trends_features, left_on='cuisine', right_on='cuisine_type', how='left')
        features = features.drop(['id', 'cuisine', 'cuisine_type'], axis=1, errors='ignore')
        
        # Add user features
        user_features = self.engineer_user_features(user_df, reviews_df)
        features = features.merge(user_features, on='restaurant_id', how='left')
        
        # Calculate target: average rating per restaurant
        restaurant_ratings = reviews_df.groupby('restaurant_id')['rating'].mean().reset_index()
        restaurant_ratings.columns = ['restaurant_id', 'avg_rating']
        
        # Merge target
        final_df = features.merge(restaurant_ratings, on='restaurant_id', how='inner')
        
        # Separate features and target
        target = final_df['avg_rating']
        feature_matrix = final_df.drop(['restaurant_id', 'avg_rating'], axis=1)
        
        # Fill NaN values
        feature_matrix = feature_matrix.fillna(0)
        
        # Store feature names
        self.feature_names = feature_matrix.columns.tolist()
        
        print(f"  - Final feature matrix: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        print(f"  - Target variable range: {target.min():.2f} - {target.max():.2f}")
        
        return feature_matrix, target
    
    def train(self, X: pd.DataFrame, y: pd.Series, tune_hyperparameters: bool = True):
        """Train the XGBoost model with optional hyperparameter tuning."""
        print("\n" + "="*80)
        print("Training Rating Prediction Model")
        print("="*80)
        
        # Split data (use larger test set for small dataset)
        test_size = 0.3 if len(X) < 100 else 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if tune_hyperparameters:
            print("\nHyperparameter tuning...")
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
            
            # Base model
            base_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
            
            # Grid search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")
            
            self.model = grid_search.best_estimator_
        else:
            # Use default parameters with regularization to prevent overfitting
            print("\nTraining with default parameters (with regularization)...")
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,  # Reduced depth to prevent overfitting
                learning_rate=0.1,
                subsample=0.8,  # Regularization
                colsample_bytree=0.8,  # Regularization
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=0.1,  # L2 regularization
                random_state=42,
                objective='reg:squarederror'
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print("\n" + "-"*80)
        print("Model Performance")
        print("-"*80)
        print(f"\nTrain Set:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  R²:   {train_r2:.4f}")
        
        print(f"\nTest Set:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  R²:   {test_r2:.4f}")
        
        self.is_trained = True
        
        return {
            'train': {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2},
            'test': {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}
        }
    
    def create_feature_matrix_for_restaurant(self, restaurant_id: int, 
                                            restaurants_df: pd.DataFrame,
                                            reviews_df: pd.DataFrame,
                                            user_df: pd.DataFrame,
                                            trends_df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for a single restaurant (reusing feature engineering pipeline)."""
        # Get restaurant data
        restaurant = restaurants_df[restaurants_df['id'] == restaurant_id]
        if restaurant.empty:
            raise ValueError(f"Restaurant {restaurant_id} not found")
        
        # Create a single-row DataFrame for this restaurant
        restaurant_subset = restaurant.copy()
        
        # Get reviews for this restaurant
        restaurant_reviews = reviews_df[reviews_df['restaurant_id'] == restaurant_id]
        
        # Engineer structured features
        structured_features = self.engineer_structured_features(restaurant_subset)
        
        # Engineer text features (use restaurant's reviews)
        if not restaurant_reviews.empty:
            text_features = self.engineer_text_features(restaurant_reviews)
        else:
            # Create a dummy review to get the correct text feature structure
            # Then we'll zero out all values
            dummy_reviews = pd.DataFrame({
                'review_id': [0],
                'restaurant_id': [restaurant_id],
                'review_text': [''],
                'rating': [0],
                'date': [pd.Timestamp.now()],
                'user_id': [0]
            })
            text_features = self.engineer_text_features(dummy_reviews)
            # Zero out all text feature values (except restaurant_id)
            for col in text_features.columns:
                if col != 'restaurant_id':
                    text_features[col] = 0.0
        
        # Merge structured and text features
        features = structured_features.merge(text_features, on='restaurant_id', how='left')
        
        # Add trend features (merge by cuisine)
        trends_features = self.engineer_timeseries_features(trends_df, reviews_df)
        features = features.merge(restaurant_subset[['id', 'cuisine']], left_on='restaurant_id', right_on='id', how='left')
        features = features.merge(trends_features, left_on='cuisine', right_on='cuisine_type', how='left')
        features = features.drop(['id', 'cuisine', 'cuisine_type'], axis=1, errors='ignore')
        
        # Add user features
        user_features = self.engineer_user_features(user_df, reviews_df)
        features = features.merge(user_features, on='restaurant_id', how='left')
        
        # Drop restaurant_id and keep only feature columns
        feature_matrix = features.drop(['restaurant_id'], axis=1, errors='ignore')
        
        # Ensure all feature columns from training are present
        for col in self.feature_names:
            if col not in feature_matrix.columns:
                feature_matrix[col] = 0.0
        
        # Reorder columns to match training order
        feature_matrix = feature_matrix[self.feature_names]
        
        # Fill NaN values
        feature_matrix = feature_matrix.fillna(0)
        
        return feature_matrix
    
    def predict(self, restaurant_id: int, restaurants_df: pd.DataFrame, 
               reviews_df: pd.DataFrame, user_df: pd.DataFrame, trends_df: pd.DataFrame) -> float:
        """Predict rating for a specific restaurant using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get restaurant data
        restaurant = restaurants_df[restaurants_df['id'] == restaurant_id]
        if restaurant.empty:
            raise ValueError(f"Restaurant {restaurant_id} not found")
        
        # Create feature matrix for this restaurant
        X = self.create_feature_matrix_for_restaurant(
            restaurant_id, restaurants_df, reviews_df, user_df, trends_df
        )
        
        # Scale features using the same scaler from training
        X_scaled = self.scaler.transform(X)
        
        # Use trained model to predict
        predicted_rating = self.model.predict(X_scaled)[0]
        
        # Clamp to valid range [1, 5]
        predicted_rating = max(1.0, min(5.0, predicted_rating))
        
        return float(predicted_rating)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for model interpretability."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def save_model(self, path: str):
        """Save model, scaler, encoders, and feature names to disk."""
        import joblib
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """Load saved model from disk."""
        import joblib
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        # Create instance
        instance = cls()
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoders = model_data['label_encoders']
        instance.tfidf_vectorizer = model_data.get('tfidf_vectorizer')
        instance.feature_names = model_data['feature_names']
        instance.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {path}")
        return instance


def main():
    """Main function to train and evaluate the model."""
    print("="*80)
    print("Task 2.1: Restaurant Rating Prediction Model")
    print("="*80)
    
    # Initialize model
    model = RatingPredictionModel()
    
    # Load data
    restaurants_df, reviews_df, user_df, trends_df = model.load_data()
    
    # Create feature matrix
    X, y = model.create_feature_matrix(restaurants_df, reviews_df, user_df, trends_df)
    
    # Train model (set tune_hyperparameters=False for faster execution)
    results = model.train(X, y, tune_hyperparameters=False)
    
    # Feature importance
    print("\n" + "="*80)
    print("Top Feature Importances")
    print("="*80)
    importance_df = model.get_feature_importance(top_n=15)
    print(importance_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Model Training Complete!")
    print("="*80)
    
    return model, results


if __name__ == "__main__":
    model, results = main()

