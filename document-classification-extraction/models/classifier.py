import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import logging
import joblib
from utils.preprocessing import extract_text, preprocess_text
from utils.feature_extraction import DocumentFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentClassifier:
    """
    Document classifier that categorizes documents into predefined classes.
    """
    
    def __init__(self, model_type='svm', max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the document classifier.
        
        Args:
            model_type (str): Type of classifier to use ('svm', 'logistic', or 'rf')
            max_features (int): Maximum number of features for the vectorizer
            ngram_range (tuple): Range of n-grams to consider
        """
        self.model_type = model_type
        self.classes = None
        self.model = None
        self.feature_extractor = DocumentFeatureExtractor(
            max_features=max_features, 
            ngram_range=ngram_range
        )
        
        # Initialize the classifier based on the model type
        if model_type == 'svm':
            self.classifier = SVC(probability=True, kernel='linear')
        elif model_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=1000, C=10.0)
        elif model_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, documents, labels, test_size=0.2, optimize=False):
        """
        Train the document classifier.
        
        Args:
            documents (list): List of document texts
            labels (list): List of document labels
            test_size (float): Proportion of data to use for testing
            optimize (bool): Whether to perform hyperparameter optimization
            
        Returns:
            dict: Training results including performance metrics
        """
        try:
            # Convert labels to numpy array
            labels = np.array(labels)
            
            # Store class labels
            self.classes = np.unique(labels)
            num_classes = len(self.classes)
            total_samples = len(documents)
            
            logger.info(f"Training with {total_samples} documents across {num_classes} classes")
            
            # Handle extreme small sample case
            if total_samples <= num_classes * 2:
                logger.warning(f"Very small dataset detected ({total_samples} samples, {num_classes} classes)")
                # Use all data for both training and testing
                X_train = documents
                y_train = labels
                X_test = documents
                y_test = labels
            else:
                # For regular case, ensure proper train/test split
                # Calculate max possible test_size
                max_test_size = (total_samples - num_classes) / total_samples
                
                # Adjust test_size if necessary
                if test_size > max_test_size:
                    adjusted_test_size = max(0.1, min(0.3, max_test_size))
                    logger.warning(f"Adjusting test_size from {test_size} to {adjusted_test_size}")
                    test_size = adjusted_test_size
                
                # Try to do a proper stratified split
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        documents, labels, test_size=test_size, random_state=42, stratify=labels
                    )
                except ValueError as e:
                    logger.warning(f"Could not perform stratified split: {str(e)}")
                    logger.warning("Using a simplified train/test approach")
                    
                    # Fallback approach: manually create train/test sets
                    # with at least one example of each class in both sets
                    X_train, X_test = [], []
                    y_train, y_test = [], []
                    
                    # For each class, put some samples in train and test
                    for c in self.classes:
                        indices = np.where(labels == c)[0]
                        if len(indices) == 1:
                            # If only one sample, use it for both train and test
                            X_train.append(documents[indices[0]])
                            y_train.append(c)
                            X_test.append(documents[indices[0]])
                            y_test.append(c)
                        else:
                            # Use 70% for training, 30% for testing (at least 1)
                            split_idx = max(1, int(len(indices) * 0.7))
                            for i in indices[:split_idx]:
                                X_train.append(documents[i])
                                y_train.append(c)
                            
                            for i in indices[split_idx:]:
                                X_test.append(documents[i])
                                y_test.append(c)
                            
                            # Ensure at least one example in test set
                            if split_idx == len(indices):
                                X_test.append(documents[indices[0]])
                                y_test.append(c)
                    
                    # Convert to arrays
                    y_train = np.array(y_train)
                    y_test = np.array(y_test)
            
            # Extract features
            logger.info("Extracting features from training documents...")
            X_train_features = self.feature_extractor.fit_transform(X_train)
            X_test_features = self.feature_extractor.transform(X_test)
            
            # Optimize hyperparameters if requested
            if optimize:
                logger.info("Optimizing hyperparameters...")
                try:
                    self._optimize_hyperparameters(X_train_features, y_train)
                except Exception as e:
                    logger.warning(f"Hyperparameter optimization failed: {str(e)}")
                    logger.warning("Using default hyperparameters instead")
            
            # Train the classifier
            logger.info(f"Training {self.model_type} classifier...")
            self.classifier.fit(X_train_features, y_train)
            
            # Evaluate on test set
            logger.info("Evaluating classifier...")
            y_pred = self.classifier.predict(X_test_features)
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store model, including the feature extractor
            self.model = {
                'classifier': self.classifier,
                'feature_extractor': self.feature_extractor,
                'classes': self.classes
            }
            
            # Return performance metrics
            return {
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'test_documents': X_test,
                'test_labels': y_test,
                'predicted_labels': y_pred
            }
        except Exception as e:
            logger.error(f"Error in training process: {str(e)}")
            raise
    
    def _optimize_hyperparameters(self, X_train_features, y_train):
        """
        Perform hyperparameter optimization using grid search.
        
        Args:
            X_train_features: Training feature vectors
            y_train: Training labels
        """
        # Count samples per class
        unique_classes, counts = np.unique(y_train, return_counts=True)
        min_samples = min(counts)
        
        # Determine cross-validation strategy based on data size
        if min_samples < 3:
            logger.warning("Too few samples per class for cross-validation")
            # Skip grid search for very small datasets
            return
        
        # Simplify parameter grid for small datasets
        if len(y_train) < 10:
            logger.warning("Small dataset detected, using minimal hyperparameter grid")
            if self.model_type == 'svm':
                param_grid = {'C': [1.0]}
            elif self.model_type == 'logistic':
                param_grid = {'C': [1.0]}
            elif self.model_type == 'rf':
                param_grid = {'n_estimators': [100]}
            cv = min(min_samples, 2)  # Use at most 2-fold CV for small datasets
        else:
            # Standard parameter grid for larger datasets
            if self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            elif self.model_type == 'logistic':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga'],
                    'penalty': ['l1', 'l2']
                }
            elif self.model_type == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            cv = min(3, min_samples)
        
        logger.info(f"Using {cv}-fold cross-validation with {len(param_grid)} parameter combinations")
        
        try:
            grid_search = GridSearchCV(
                self.classifier, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1
            )
            grid_search.fit(X_train_features, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            self.classifier = grid_search.best_estimator_
        except Exception as e:
            logger.warning(f"Grid search failed: {str(e)}")
            logger.warning("Using default parameters instead")
    
    def predict(self, document):
        """
        Predict the class of a document.
        
        Args:
            document (str): Document text
            
        Returns:
            dict: Prediction results including class, confidence, and top features
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess the document
        processed_doc = preprocess_text(document)
        
        # Extract features
        features = self.feature_extractor.transform([processed_doc])
        
        # Make prediction
        predicted_class_idx = self.classifier.predict(features)[0]
        predicted_class = self.classes[predicted_class_idx] if isinstance(predicted_class_idx, (int, np.integer)) else predicted_class_idx
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(features)[0]
        confidence = max(probabilities)
        
        # Get confidence for each class
        class_confidences = {}
        for i, class_label in enumerate(self.classifier.classes_):
            if isinstance(class_label, (int, np.integer)):
                class_name = self.classes[class_label]
            else:
                class_name = class_label
            class_confidences[class_name] = float(probabilities[i])
        
        # Get top features for the prediction - disabled for now due to issues with sparse arrays
        # Just return an empty list for top features
        top_features = []
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'class_confidences': class_confidences,
            'top_features': top_features
        }
    
    def save(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            DocumentClassifier: Loaded document classifier
        """
        try:
            # Load the model
            model_data = joblib.load(filepath)
            
            # Create a new instance
            instance = cls()
            
            # Set the loaded model components
            instance.classifier = model_data['classifier']
            instance.feature_extractor = model_data['feature_extractor']
            instance.classes = model_data['classes']
            instance.model = model_data
            
            logger.info(f"Model loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, y_true, y_pred, classes=None, figsize=(10, 8)):
        """
        Plot the confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes (list): List of class names (optional)
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if classes is None:
            classes = self.classes
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Show all ticks
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='True label',
            xlabel='Predicted label'
        )
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig