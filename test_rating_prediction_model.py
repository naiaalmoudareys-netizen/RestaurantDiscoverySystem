"""
Unit Tests for Rating Prediction Model (Task 2.1)
Tests individual functions and methods in isolation using mocks.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from rating_prediction_model import RatingPredictionModel


class TestRatingPredictionModel:
    """Unit tests for RatingPredictionModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance."""
        return RatingPredictionModel()
    
    @pytest.fixture
    def sample_restaurants_df(self):
        """Sample restaurants DataFrame."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
            'cuisine': ['Italian', 'Chinese', 'Italian'],
            'location': ['Downtown', 'Palm', 'Downtown'],
            'price_range': ['AED 100-200', 'AED 50-100', 'AED 200-300'],
            'rating': [4.5, 4.2, 4.8],
            'review_count': [150, 200, 100],
            'amenities': ['WiFi, Parking, Outdoor Seating', 'WiFi, Parking', 'WiFi, Live Music'],
            'attributes': ['Romantic, Family-friendly', 'Casual', 'Fine Dining, Romantic']
        })
    
    @pytest.fixture
    def sample_reviews_df(self):
        """Sample reviews DataFrame."""
        return pd.DataFrame({
            'review_id': [1, 2, 3, 4, 5],
            'restaurant_id': [1, 1, 2, 2, 3],
            'user_id': [1, 2, 1, 3, 2],
            'rating': [5.0, 4.0, 4.5, 4.0, 5.0],
            'review_text': [
                'Great food!',
                'Good service',
                'Amazing experience',
                'Nice ambiance',
                'Excellent!'
            ],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        })
    
    @pytest.fixture
    def sample_user_df(self):
        """Sample user DataFrame."""
        return pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [30, 25, 35],
            'dining_frequency': ['Weekly', 'Monthly', 'Weekly'],
            'avg_rating_given': [4.5, 4.0, 4.8],
            'total_reviews_written': [50, 20, 100]
        })
    
    @pytest.fixture
    def sample_trends_df(self):
        """Sample trends DataFrame."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5, freq='D'),
            'cuisine_type': ['Italian', 'Chinese', 'Italian', 'Chinese', 'Italian'],
            'popularity_score': [0.8, 0.7, 0.9, 0.6, 0.85],
            'avg_price': [150, 75, 160, 80, 155],
            'booking_lead_time_days': [3, 2, 4, 1, 3],
            'season': ['Peak', 'Off-Peak', 'Peak', 'Off-Peak', 'Peak'],
            'day_type': ['Weekend', 'Weekday', 'Weekend', 'Weekday', 'Weekend'],
            'is_holiday': [1, 0, 1, 0, 1],
            'weather_impact_category': ['Favorable Weather', 'Neutral', 'Favorable Weather', 
                                        'Neutral', 'Favorable Weather']
        })
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.model is None
        assert model.is_trained is False
        assert model.feature_names == []
        assert model.label_encoders == {}
    
    def test_engineer_structured_features(self, model, sample_restaurants_df):
        """Test structured feature engineering."""
        features = model.engineer_structured_features(sample_restaurants_df)
        
        assert 'restaurant_id' in features.columns
        assert 'cuisine_encoded' in features.columns
        assert 'location_encoded' in features.columns
        assert len(features) == 3
        assert len(features.columns) > 3  # Should have multiple features
    
    def test_engineer_structured_features_encoders(self, model, sample_restaurants_df):
        """Test that label encoders are created and stored."""
        features1 = model.engineer_structured_features(sample_restaurants_df)
        
        # Encoders should be stored
        assert 'cuisine' in model.label_encoders
        assert 'location' in model.label_encoders
        
        # Second call should use existing encoders
        features2 = model.engineer_structured_features(sample_restaurants_df)
        
        # Encoded values should be consistent
        assert (features1['cuisine_encoded'] == features2['cuisine_encoded']).all()
    
    def test_engineer_text_features(self, model, sample_reviews_df):
        """Test text feature engineering from reviews."""
        features = model.engineer_text_features(sample_reviews_df)
        
        assert 'restaurant_id' in features.columns
        assert len(features) > 0
        # Should have aggregated review statistics
        assert any('avg' in col.lower() or 'count' in col.lower() or 'sentiment' in col.lower() 
                  for col in features.columns)
    
    def test_engineer_timeseries_features(self, model, sample_trends_df, sample_reviews_df):
        """Test time series feature engineering."""
        features = model.engineer_timeseries_features(sample_trends_df, sample_reviews_df)
        
        assert 'cuisine_type' in features.columns
        assert len(features) > 0
        # Should have trend-related features
        assert any('trend' in col.lower() or 'popularity' in col.lower() 
                  for col in features.columns)
    
    def test_engineer_user_features(self, model, sample_user_df, sample_reviews_df):
        """Test user feature engineering."""
        features = model.engineer_user_features(sample_user_df, sample_reviews_df)
        
        assert 'restaurant_id' in features.columns
        assert len(features) > 0
        # Should have user-related features
        assert any('age' in col.lower() or 'frequency' in col.lower() or 'rating' in col.lower()
                  for col in features.columns)
    
    def test_create_feature_matrix(self, model, sample_restaurants_df, sample_reviews_df, 
                                   sample_user_df, sample_trends_df):
        """Test complete feature matrix creation."""
        X, y = model.create_feature_matrix(
            sample_restaurants_df, sample_reviews_df, sample_user_df, sample_trends_df
        )
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X.columns) > 0
        assert model.feature_names == list(X.columns)
        # Target should be ratings
        assert y.min() >= 1.0
        assert y.max() <= 5.0
    
    def test_create_feature_matrix_handles_missing_data(self, model, sample_restaurants_df, 
                                                        sample_reviews_df, sample_user_df, 
                                                        sample_trends_df):
        """Test that feature matrix handles missing data."""
        # Add some missing data
        sample_reviews_df.loc[0, 'rating'] = np.nan
        
        X, y = model.create_feature_matrix(
            sample_restaurants_df, sample_reviews_df, sample_user_df, sample_trends_df
        )
        
        # Should handle NaN values (fill with 0 or drop)
        assert not X.isnull().any().any() or X.isnull().sum().sum() == 0
    
    @patch('rating_prediction_model.xgb.XGBRegressor')
    def test_train(self, mock_xgb, model):
        """Test model training."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.5, 0.6, 0.7, 0.8, 0.9]
        })
        y = pd.Series([4.5, 4.2, 4.8, 4.0, 4.6])
        
        mock_model = Mock()
        mock_model.predict.return_value = y.values
        mock_xgb.return_value = mock_model
        
        # Mock train_test_split
        with patch('rating_prediction_model.train_test_split') as mock_split:
            mock_split.return_value = (X, X, y, y)  # Same data for train/test
            
            results = model.train(X, y, tune_hyperparameters=False)
            
            assert model.is_trained is True
            assert 'train' in results
            assert 'test' in results
            assert 'rmse' in results['train']
            assert 'mae' in results['train']
            assert 'r2' in results['train']
    
    def test_create_feature_matrix_for_restaurant(self, model, sample_restaurants_df,
                                                   sample_reviews_df, sample_user_df,
                                                   sample_trends_df):
        """Test creating feature matrix for a single restaurant."""
        # First create full feature matrix to set up encoders
        X, y = model.create_feature_matrix(
            sample_restaurants_df, sample_reviews_df, sample_user_df, sample_trends_df
        )
        
        # Now test single restaurant feature creation
        features = model.create_feature_matrix_for_restaurant(
            1, sample_restaurants_df, sample_reviews_df, sample_user_df, sample_trends_df
        )
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1
        assert len(features.columns) == len(X.columns)
    
    def test_get_feature_importance_untrained(self, model):
        """Test getting feature importance when model is not trained."""
        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_feature_importance()
    
    def test_save_and_load_model(self, model, sample_restaurants_df, sample_reviews_df,
                                 sample_user_df, sample_trends_df):
        """Test saving and loading model."""
        # Create and train a simple model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.5, 0.6, 0.7, 0.8, 0.9]
        })
        y = pd.Series([4.5, 4.2, 4.8, 4.0, 4.6])
        
        with patch('rating_prediction_model.train_test_split') as mock_split:
            mock_split.return_value = (X, X, y, y)
            
            model.train(X, y, tune_hyperparameters=False)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            model_path = tmp_file.name
        
        try:
            model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = RatingPredictionModel.load_model(model_path)
            assert loaded_model.is_trained is True
            assert loaded_model.feature_names == model.feature_names
            
        finally:
            # Cleanup
            if os.path.exists(model_path):
                os.remove(model_path)


class TestRatingPredictionModelEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def model(self):
        return RatingPredictionModel()
    
    def test_engineer_structured_features_empty_df(self, model):
        """Test structured features with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises((KeyError, IndexError)):
            model.engineer_structured_features(empty_df)
    
    def test_engineer_text_features_empty_reviews(self, model):
        """Test text features with empty reviews."""
        empty_df = pd.DataFrame(columns=['review_id', 'restaurant_id', 'rating', 'review_text'])
        features = model.engineer_text_features(empty_df)
        
        # Should return empty DataFrame or handle gracefully
        assert isinstance(features, pd.DataFrame)
    
    def test_create_feature_matrix_no_matching_data(self, model):
        """Test feature matrix creation with no matching data."""
        restaurants_df = pd.DataFrame({'id': [1], 'cuisine': ['Italian']})
        reviews_df = pd.DataFrame({'restaurant_id': [999], 'rating': [4.5]})
        user_df = pd.DataFrame({'user_id': [1]})
        trends_df = pd.DataFrame({'cuisine_type': ['French']})
        
        # Should handle gracefully (may return empty or raise error)
        try:
            X, y = model.create_feature_matrix(restaurants_df, reviews_df, user_df, trends_df)
            # If it succeeds, check that it handles missing data
            assert isinstance(X, pd.DataFrame)
        except (KeyError, ValueError):
            # Expected if data doesn't match
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

