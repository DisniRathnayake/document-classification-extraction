"""
Feature extraction utilities for document classification.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentFeatureExtractor:
    """
    Extract features from documents for classification.
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the feature extractor.
        
        Args:
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): Range of n-grams to consider
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, documents):
        """
        Fit the vectorizer on a collection of documents.
        
        Args:
            documents (list): List of document texts
        """
        try:
            self.vectorizer.fit(documents)
            self.is_fitted = True
            logger.info(f"Fitted TF-IDF vectorizer with {len(self.vectorizer.get_feature_names_out())} features")
            return self
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {str(e)}")
            raise
    
    def transform(self, documents):
        """
        Transform documents to feature vectors.
        
        Args:
            documents (list): List of document texts
            
        Returns:
            sparse matrix: Document feature vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer is not fitted. Call fit() first.")
        
        try:
            return self.vectorizer.transform(documents)
        except Exception as e:
            logger.error(f"Error transforming documents: {str(e)}")
            raise
    
    def fit_transform(self, documents):
        """
        Fit the vectorizer and transform documents in one step.
        
        Args:
            documents (list): List of document texts
            
        Returns:
            sparse matrix: Document feature vectors
        """
        try:
            features = self.vectorizer.fit_transform(documents)
            self.is_fitted = True
            logger.info(f"Fitted TF-IDF vectorizer with {len(self.vectorizer.get_feature_names_out())} features")
            return features
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def get_feature_names(self):
        """
        Get the names of the features (vocabulary).
        
        Returns:
            list: Feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer is not fitted. Call fit() first.")
        
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features_per_class(self, classifier, class_labels, n=20):
        """
        Get the top features for each class based on the classifier coefficients.
        
        Args:
            classifier: Trained classifier with coef_ attribute
            class_labels (list): List of class labels
            n (int): Number of top features to return per class
            
        Returns:
            dict: Dictionary mapping class labels to top feature words
        """
        if not hasattr(classifier, 'coef_'):
            logger.warning("Classifier doesn't have coef_ attribute. Can't extract top features.")
            return {}
        
        feature_names = self.get_feature_names()
        top_features = {}
        
        # For binary classification
        if len(class_labels) == 2 and classifier.coef_.shape[0] == 1:
            # Positive coefficients for class 1, negative for class 0
            top_indices = np.argsort(classifier.coef_[0])
            
            # Top features for class 0 (negative coefficients)
            top_features[class_labels[0]] = [feature_names[i] for i in top_indices[:n]]
            
            # Top features for class 1 (positive coefficients)
            top_features[class_labels[1]] = [feature_names[i] for i in top_indices[-n:]]
        
        # For multiclass
        else:
            for i, label in enumerate(class_labels):
                top_indices = np.argsort(classifier.coef_[i])[-n:]
                top_features[label] = [feature_names[j] for j in top_indices]
        
        return top_features