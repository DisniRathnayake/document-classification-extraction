"""
Unit tests for the document classification model.
"""

import os
import unittest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from models.classifier import DocumentClassifier
from utils.feature_extraction import DocumentFeatureExtractor

class TestDocumentClassifier(unittest.TestCase):
    """Test cases for document classifier."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Sample documents for testing
        self.sample_documents = [
            "This is an invoice for $500. Invoice #12345. Payment due in 30 days.",
            "Professional resume with 10 years of experience in software development.",
            "This contract is between Party A and Party B for services rendered.",
            "Invoice from XYZ Corp. Total amount: $1,200.50",
            "Resume: Skills include Python, JavaScript, and project management.",
            "Service agreement contract with termination clause and payment terms."
        ]
        
        # Sample labels
        self.sample_labels = ['invoice', 'resume', 'contract', 'invoice', 'resume', 'contract']
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test classifier initialization with different models."""
        # Test default initialization (SVM)
        classifier = DocumentClassifier()
        self.assertEqual(classifier.model_type, 'svm')
        self.assertIsNone(classifier.model)
        self.assertIsNone(classifier.classes)
        self.assertIsInstance(classifier.feature_extractor, DocumentFeatureExtractor)
        
        # Test initialization with logistic regression
        classifier = DocumentClassifier(model_type='logistic')
        self.assertEqual(classifier.model_type, 'logistic')
        self.assertIsInstance(classifier.classifier, LogisticRegression)
        
        # Test initialization with random forest
        classifier = DocumentClassifier(model_type='rf')
        self.assertEqual(classifier.model_type, 'rf')
        
        # Test initialization with invalid model type
        with self.assertRaises(ValueError):
            DocumentClassifier(model_type='invalid_model')
    
    def test_train_predict_basic(self):
        """Test basic training and prediction functionality."""
        # Create and train classifier
        classifier = DocumentClassifier(model_type='logistic')
        
        # Train on sample data
        results = classifier.train(
            documents=self.sample_documents,
            labels=self.sample_labels,
            test_size=0.5,
            optimize=False
        )
        
        # Check if model is created
        self.assertIsNotNone(classifier.model)
        self.assertIsNotNone(classifier.classes)
        
        # Check if expected results are returned
        self.assertIn('classification_report', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('test_documents', results)
        self.assertIn('test_labels', results)
        self.assertIn('predicted_labels', results)
        
        # Test prediction on new document
        new_document = "Invoice #5678 for services rendered. Total: $750."
        prediction = classifier.predict(new_document)
        
        # Check prediction structure
        self.assertIn('predicted_class', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('class_confidences', prediction)
        self.assertIn('top_features', prediction)
        
        # Check if confidence is a float between 0 and 1
        self.assertIsInstance(prediction['confidence'], float)
        self.assertTrue(0 <= prediction['confidence'] <= 1)
        
        # Check if class_confidences contains all classes
        self.assertEqual(set(prediction['class_confidences'].keys()), set(self.sample_labels))
        
        # Check if top_features is a list
        self.assertIsInstance(prediction['top_features'], list)
        
        # Check if predictions match a reasonable class, most likely 'invoice'
        self.assertEqual(prediction['predicted_class'], 'invoice')
    
    @patch('joblib.dump')
    @patch('joblib.load')
    def test_save_load(self, mock_load, mock_dump):
        """Test saving and loading the model."""
        # Train a classifier
        classifier = DocumentClassifier(model_type='logistic')
        classifier.train(
            documents=self.sample_documents,
            labels=self.sample_labels,
            test_size=0.5,
            optimize=False
        )
        
        # Set up mock for loading
        mock_load.return_value = {
            'classifier': classifier.classifier,
            'feature_extractor': classifier.feature_extractor,
            'classes': np.array(['invoice', 'resume', 'contract'])
        }
        
        # Test saving the model
        model_path = os.path.join(self.temp_dir.name, "test_model.joblib")
        classifier.save(model_path)
        mock_dump.assert_called_once()
        
        # Test loading the model
        loaded_classifier = DocumentClassifier.load(model_path)
        mock_load.assert_called_once_with(model_path)
        
        # Check if loaded model has the same classes
        np.testing.assert_array_equal(loaded_classifier.classes, classifier.classes)
    
    def test_predict_without_training(self):
        """Test prediction without training should raise ValueError."""
        classifier = DocumentClassifier()
        with self.assertRaises(ValueError):
            classifier.predict("This is a test document.")
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_confusion_matrix(self, mock_subplots):
        """Test confusion matrix plotting functionality."""
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Create classifier
        classifier = DocumentClassifier()
        classifier.classes = np.array(['invoice', 'resume', 'contract'])
        
        # Test data
        y_true = ['invoice', 'resume', 'contract', 'invoice', 'resume', 'contract']
        y_pred = ['invoice', 'invoice', 'contract', 'invoice', 'resume', 'resume']
        
        # Plot confusion matrix
        fig = classifier.plot_confusion_matrix(y_true, y_pred)
        
        # Check that the figure is returned
        self.assertEqual(fig, mock_fig)
        
        # Check that the necessary method calls were made
        mock_ax.imshow.assert_called_once()
        self.assertEqual(mock_ax.set.call_count, 1)
    
    def test_optimize_hyperparameters(self):
        """Test hyperparameter optimization for different models."""
        # Sample feature matrix (simplified for testing)
        X_train_features = np.random.rand(6, 10)
        y_train = np.array(self.sample_labels)
        
        # Test SVM optimization
        with patch('models.classifier.GridSearchCV') as mock_grid_search:
            mock_grid_search_instance = MagicMock()
            mock_grid_search.return_value = mock_grid_search_instance
            
            classifier = DocumentClassifier(model_type='svm')
            classifier._optimize_hyperparameters(X_train_features, y_train)
            
            # Check if GridSearchCV was called with appropriate parameters
            mock_grid_search.assert_called_once()
            mock_grid_search_instance.fit.assert_called_once_with(X_train_features, y_train)
        
        # Test Logistic Regression optimization
        with patch('models.classifier.GridSearchCV') as mock_grid_search:
            mock_grid_search_instance = MagicMock()
            mock_grid_search.return_value = mock_grid_search_instance
            
            classifier = DocumentClassifier(model_type='logistic')
            classifier._optimize_hyperparameters(X_train_features, y_train)
            
            # Check if GridSearchCV was called with appropriate parameters
            mock_grid_search.assert_called_once()
            mock_grid_search_instance.fit.assert_called_once_with(X_train_features, y_train)
        
        # Test Random Forest optimization
        with patch('models.classifier.GridSearchCV') as mock_grid_search:
            mock_grid_search_instance = MagicMock()
            mock_grid_search.return_value = mock_grid_search_instance
            
            classifier = DocumentClassifier(model_type='rf')
            classifier._optimize_hyperparameters(X_train_features, y_train)
            
            # Check if GridSearchCV was called with appropriate parameters
            mock_grid_search.assert_called_once()
            mock_grid_search_instance.fit.assert_called_once_with(X_train_features, y_train)

if __name__ == '__main__':
    unittest.main()