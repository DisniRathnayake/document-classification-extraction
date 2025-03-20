"""
Main Streamlit application for document classification and data extraction.
"""

import os
import streamlit as st
import pandas as pd
import tempfile
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import logging
import time
import base64
from io import BytesIO

# Import custom modules
from utils.preprocessing import extract_text, preprocess_text, extract_named_entities
from models.classifier import DocumentClassifier
from models.extractor import DataExtractor
from utils.validation import DocumentValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Document Classifier & Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = 'models/saved/document_classifier.joblib'
EXTRACTION_CONFIG_PATH = 'models/saved/extraction_config.json'
TEMP_UPLOAD_PATH = 'temp_uploads'
SAMPLE_DOCS_PATH = 'data/sample_documents'

# Ensure directories exist
os.makedirs('models/saved', exist_ok=True)
os.makedirs(TEMP_UPLOAD_PATH, exist_ok=True)

def load_or_train_model():
    """
    Load the trained model if it exists, otherwise train a new one.
    
    Returns:
        DocumentClassifier: Trained document classifier
    """
    if os.path.exists(MODEL_PATH):
        try:
            st.sidebar.info("Loading pre-trained classification model...")
            return DocumentClassifier.load(MODEL_PATH)
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
    
    st.sidebar.warning("No pre-trained model found. Please train a new model.")
    return None

def get_document_types():
    """
    Get available document types.
    
    Returns:
        list: List of document types
    """
    return ['invoice', 'resume', 'contract']

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to the saved file
    """
    try:
        # Create the temporary directory if it doesn't exist
        os.makedirs(TEMP_UPLOAD_PATH, exist_ok=True)
        
        # Create a temporary file path
        file_path = os.path.join(TEMP_UPLOAD_PATH, uploaded_file.name)
        
        # Write the file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def display_extracted_data(extracted_data):
    """
    Display extracted data in a formatted way.
    
    Args:
        extracted_data (dict): Extracted data
    """
    # Display document type
    st.subheader(f"Document Type: {extracted_data.get('document_type', 'Unknown')}")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Extracted Fields", "Named Entities"])
    
    with tab1:
        # Display main extracted fields
        fields_to_display = {k: v for k, v in extracted_data.items() 
                           if not k.startswith('entities_') and k != 'document_type'}
        
        if not fields_to_display:
            st.info("No fields extracted.")
        else:
            for field, value in fields_to_display.items():
                if field == 'line_items' and isinstance(value, list):
                    st.write("**Line Items:**")
                    # Convert to DataFrame for better display
                    items_df = pd.DataFrame(value)
                    st.dataframe(items_df)
                elif field == 'experience_sections' and isinstance(value, list):
                    st.write("**Work Experience:**")
                    # Convert to DataFrame for better display
                    exp_df = pd.DataFrame(value)
                    st.dataframe(exp_df)
                elif field == 'skills_list' and isinstance(value, list):
                    st.write("**Skills:**")
                    # Display as pills/tags
                    cols = st.columns(4)
                    for i, skill in enumerate(value):
                        cols[i % 4].markdown(f"<span style='background-color: #f0f2f6; padding: 5px 10px; border-radius: 15px; margin: 2px; display: inline-block;'>{skill}</span>", unsafe_allow_html=True)
                else:
                    st.write(f"**{field.replace('_', ' ').title()}:** {value}")
    
    with tab2:
        # Display named entities
        entities = {k: v for k, v in extracted_data.items() if k.startswith('entities_')}
        
        if not entities:
            st.info("No named entities extracted.")
        else:
            for entity_type, values in entities.items():
                # Get entity type name from the key (e.g., entities_PERSON -> PERSON)
                entity_name = entity_type.split('_')[1]
                st.write(f"**{entity_name}:**")
                
                # Display entities as pills/tags
                cols = st.columns(4)
                for i, value in enumerate(values):
                    cols[i % 4].markdown(f"<span style='background-color: #f0f2f6; padding: 5px 10px; border-radius: 15px; margin: 2px; display: inline-block;'>{value}</span>", unsafe_allow_html=True)

def train_model_ui():
    """
    UI for training a new classification model.
    """
    st.header("Train Document Classification Model")
    
    # Input for training data
    st.write("### Training Data")
    st.write("Upload document examples for each category for training:")
    
    # Create columns for each document type
    doc_types = get_document_types()
    cols = st.columns(len(doc_types))
    
    training_files = {}
    for i, doc_type in enumerate(doc_types):
        with cols[i]:
            st.write(f"**{doc_type.title()} Documents**")
            uploaded_files = st.file_uploader(
                f"Upload {doc_type} documents",
                type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key=f"train_{doc_type}"
            )
            if uploaded_files:
                training_files[doc_type] = uploaded_files
    
    # Model configuration
    st.write("### Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Classification Algorithm",
            options=['svm', 'logistic', 'rf'],
            format_func=lambda x: {
                'svm': 'Support Vector Machine',
                'logistic': 'Logistic Regression',
                'rf': 'Random Forest'
            }.get(x, x)
        )
        
        optimize = st.checkbox("Optimize Hyperparameters", value=True)
    
    with col2:
        max_features = st.slider("Maximum Features", min_value=1000, max_value=10000, value=5000, step=1000)
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    # Train button
    train_button = st.button("Train Model", type="primary")
    
    if train_button:
        if not any(training_files.values()):
            st.error("Please upload at least one document for each category.")
            return
        
        # Check if there are at least some files for each document type
        if not all(training_files.get(doc_type) for doc_type in doc_types):
            st.warning("Some document types have no training examples. This may affect model performance.")
        
        # Process uploaded files
        with st.spinner("Processing training documents..."):
            documents = []
            labels = []
            
            for doc_type, files in training_files.items():
                for file in files:
                    try:
                        # Save the uploaded file
                        file_path = save_uploaded_file(file)
                        
                        if file_path:
                            # Extract and preprocess text
                            text = extract_text(file_path)
                            processed_text = preprocess_text(text)
                            
                            # Add to training data
                            documents.append(processed_text)
                            labels.append(doc_type)
                    except Exception as e:
                        st.error(f"Error processing file {file.name}: {str(e)}")
        
        # Train the model
        with st.spinner("Training classification model..."):
            try:
                # Create and train classifier
                classifier = DocumentClassifier(
                    model_type=model_type,
                    max_features=max_features
                )
                
                results = classifier.train(
                    documents=documents,
                    labels=labels,
                    test_size=test_size,
                    optimize=optimize
                )
                
                # Save the model
                classifier.save(MODEL_PATH)
                
                # Display training results
                st.success("Model trained successfully!")
                
                # Display classification report
                st.write("### Performance Metrics")
                st.write("Classification Report:")
                report_df = pd.DataFrame(results['classification_report']).transpose()
                st.dataframe(report_df)
                
                # Plot confusion matrix
                st.write("Confusion Matrix:")
                fig = classifier.plot_confusion_matrix(
                    results['test_labels'],
                    results['predicted_labels']
                )
                st.pyplot(fig)
                
                # Show some examples
                st.write("### Prediction Examples")
                st.write("Here are some example predictions from the test set:")
                
                # Get a few examples from each class
                examples_per_class = 2
                example_indices = {}
                for i, label in enumerate(results['test_labels']):
                    if label not in example_indices:
                        example_indices[label] = []
                    
                    if len(example_indices[label]) < examples_per_class:
                        example_indices[label].append(i)
                
                # Display examples
                for label, indices in example_indices.items():
                    st.write(f"**Category: {label}**")
                    
                    for i in indices:
                        # Get prediction for this document
                        prediction = classifier.predict(results['test_documents'][i])
                        
                        # Display in expandable section
                        with st.expander(f"Document {i+1} - Predicted: {prediction['predicted_class']} (Confidence: {prediction['confidence']:.2f})"):
                            # Show top features
                            st.write("Top Features:")
                            for feature in prediction['top_features'][:5]:
                                st.markdown(f"- {feature}")
                            
                            # Show document text (truncated)
                            text = results['test_documents'][i]
                            st.text_area("Document Text (excerpt)", text[:500] + ("..." if len(text) > 500 else ""), height=100)
            
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                logger.error(f"Error training model: {str(e)}", exc_info=True)

def classify_document_ui():
    """
    UI for classifying and extracting data from documents.
    """
    st.header("Document Classification and Data Extraction")
    
    # Load model
    classifier = load_or_train_model()
    
    if classifier is None:
        st.info("Please train a classification model first.")
        return
    
    # Load data extractor
    data_extractor = DataExtractor()
    
    # Upload document
    st.write("### Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a document for classification and data extraction",
        type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        # Save the uploaded file
        with st.spinner("Processing document..."):
            file_path = save_uploaded_file(uploaded_file)
            
            if file_path:
                try:
                    # Extract text
                    text = extract_text(file_path)
                    
                    if not text:
                        st.error("No text could be extracted from the document.")
                        return
                    
                    # Display extracted text in an expander
                    with st.expander("View Extracted Text"):
                        st.text_area("Text", text, height=200)
                    
                    # Preprocess text for classification
                    processed_text = preprocess_text(text)
                    
                    # Classify document
                    prediction = classifier.predict(processed_text)
                    
                    # Display classification results
                    st.write("### Classification Results")
                    
                    # Create a progress bar for confidence
                    st.write(f"**Predicted Document Type:** {prediction['predicted_class'].title()}")
                    st.write("**Confidence:**")
                    st.progress(prediction['confidence'])
                    st.write(f"{prediction['confidence']:.2f}")
                    
                    # Display confidence for each class
                    st.write("**Class Probabilities:**")
                    for class_name, confidence in prediction['class_confidences'].items():
                        st.write(f"{class_name.title()}: {confidence:.2f}")
                    
                    # Display top features
                    st.write("**Top Features:**")
                    for feature in prediction['top_features'][:10]:
                        st.markdown(f"- {feature}")
                    
                    # Extract data based on classified document type
                    st.write("### Extracted Data")
                    with st.spinner("Extracting data..."):
                        extracted_data = data_extractor.extract_data(text, prediction['predicted_class'])
                        
                        # Validate extracted data
                        validation_result = DocumentValidator.validate_extracted_data(
                            extracted_data, prediction['predicted_class']
                        )
                        
                        # Display extracted data
                        display_extracted_data(extracted_data)
                        
                        # Display validation results
                        if not validation_result['is_valid']:
                            st.warning("**Validation Issues:**")
                            
                            if validation_result['missing_fields']:
                                st.write("Missing Fields:")
                                for field in validation_result['missing_fields']:
                                    st.markdown(f"- {field}")
                            
                            if validation_result['invalid_fields']:
                                st.write("Invalid Fields:")
                                for field in validation_result['invalid_fields']:
                                    st.markdown(f"- {field}")
                
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    logger.error(f"Error processing document: {str(e)}", exc_info=True)

def extraction_config_ui():
    """
    UI for configuring data extraction patterns.
    """
    st.header("Data Extraction Configuration")
    
    # Load data extractor
    data_extractor = DataExtractor()
    
    # Display current configuration
    st.write("### Current Configuration")
    
    # Create tabs for each document type
    doc_types = get_document_types()
    tabs = st.tabs([doc_type.title() for doc_type in doc_types])
    
    # Display and edit configuration for each document type
    for i, doc_type in enumerate(doc_types):
        with tabs[i]:
            if doc_type in data_extractor.config:
                # Display patterns
                st.write("**Extraction Patterns:**")
                
                patterns = data_extractor.config[doc_type].get('patterns', {})
                patterns_dict = {}
                
                for field, pattern in patterns.items():
                    patterns_dict[field] = {"pattern": pattern, "edit": False}
                
                # Create a form for each pattern
                for field, data in patterns_dict.items():
                    with st.expander(f"{field.replace('_', ' ').title()}"):
                        # Display current pattern
                        st.code(data['pattern'], language='regex')
                        
                        # Edit pattern
                        new_pattern = st.text_input(f"Update pattern for {field}", data['pattern'])
                        
                        # Update pattern if changed
                        if new_pattern != data['pattern']:
                            data_extractor.add_extraction_pattern(doc_type, field, new_pattern)
                            st.success(f"Updated pattern for {field}")
                
                # Add new pattern
                st.write("**Add New Pattern:**")
                with st.form(f"new_pattern_{doc_type}"):
                    new_field = st.text_input("Field Name")
                    new_pattern = st.text_input("Regular Expression Pattern")
                    submit = st.form_submit_button("Add Pattern")
                    
                    if submit and new_field and new_pattern:
                        data_extractor.add_extraction_pattern(doc_type, new_field, new_pattern)
                        st.success(f"Added new pattern for {new_field}")
                        st.experimental_rerun()
                
                # Display entities
                st.write("**Named Entities to Extract:**")
                
                entities = data_extractor.config[doc_type].get('entities', [])
                
                if entities:
                    cols = st.columns(4)
                    for i, entity in enumerate(entities):
                        cols[i % 4].markdown(f"<span style='background-color: #f0f2f6; padding: 5px 10px; border-radius: 15px; margin: 2px; display: inline-block;'>{entity}</span>", unsafe_allow_html=True)
                
                # Add new entity
                st.write("**Add New Entity Type:**")
                
                # Common spaCy entity types
                spacy_entities = [
                    'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 
                    'DATE', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY', 'ORDINAL', 'CARDINAL'
                ]
                
                new_entity = st.selectbox("Entity Type", options=[e for e in spacy_entities if e not in entities])
                add_entity = st.button(f"Add {new_entity}", key=f"add_entity_{doc_type}")
                
                if add_entity:
                    data_extractor.add_entity_type(doc_type, new_entity)
                    st.success(f"Added entity type {new_entity}")
                    st.experimental_rerun()
            
            else:
                st.warning(f"No configuration found for {doc_type}")
    
    # Save configuration
    st.write("### Save Configuration")
    
    save_config = st.button("Save Configuration")
    
    if save_config:
        try:
            # Save the configuration
            data_extractor.save_config(EXTRACTION_CONFIG_PATH)
            st.success("Configuration saved successfully!")
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")
            logger.error(f"Error saving configuration: {str(e)}", exc_info=True)

def batch_processing_ui():
    """
    UI for batch processing multiple documents.
    """
    st.header("Batch Document Processing")
    
    # Load model
    classifier = load_or_train_model()
    
    if classifier is None:
        st.info("Please train a classification model first.")
        return
    
    # Load data extractor
    data_extractor = DataExtractor()
    
    # Upload documents
    st.write("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload multiple documents for batch processing",
        type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Process batch
        process_batch = st.button("Process Batch", type="primary")
        
        if process_batch:
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Initialize results container
            results = []
            
            # Process each file
            for i, file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Status message
                    st.write(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
                    
                    # Save the uploaded file
                    file_path = save_uploaded_file(file)
                    
                    if file_path:
                        # Extract text
                        text = extract_text(file_path)
                        
                        if not text:
                            st.warning(f"No text could be extracted from {file.name}.")
                            continue
                        
                        # Preprocess text for classification
                        processed_text = preprocess_text(text)
                        
                        # Classify document
                        prediction = classifier.predict(processed_text)
                        
                        # Extract data based on classified document type
                        extracted_data = data_extractor.extract_data(text, prediction['predicted_class'])
                        
                        # Validate extracted data
                        validation_result = DocumentValidator.validate_extracted_data(
                            extracted_data, prediction['predicted_class']
                        )
                        
                        # Add to results
                        results.append({
                            'filename': file.name,
                            'document_type': prediction['predicted_class'],
                            'confidence': prediction['confidence'],
                            'is_valid': validation_result['is_valid'],
                            'missing_fields': validation_result['missing_fields'],
                            'invalid_fields': validation_result['invalid_fields'],
                            'extracted_data': extracted_data
                        })
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
            
            # Complete the progress bar
            progress_bar.progress(1.0)
            
            # Display results
            st.write("### Batch Processing Results")
            
            # Create a summary table
            summary_data = []
            for result in results:
                summary_data.append({
                    'Filename': result['filename'],
                    'Document Type': result['document_type'].title(),
                    'Confidence': f"{result['confidence']:.2f}",
                    'Valid': "✓" if result['is_valid'] else "✗",
                    'Issues': len(result['missing_fields']) + len(result['invalid_fields'])
                })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data))
            
            # Download results
            if results:
                # Convert results to JSON
                results_json = json.dumps(results, indent=2)
                
                # Create a download button
                st.download_button(
                    label="Download Results (JSON)",
                    data=results_json,
                    file_name="batch_results.json",
                    mime="application/json"
                )
                
                # Option to view detailed results
                st.write("### Detailed Results")
                for i, result in enumerate(results):
                    with st.expander(f"{result['filename']} ({result['document_type'].title()})"):
                        # Display classification info
                        st.write(f"**Document Type:** {result['document_type'].title()}")
                        st.write(f"**Confidence:** {result['confidence']:.2f}")
                        
                        # Display validation info
                        st.write("**Validation:**")
                        if result['is_valid']:
                            st.success("All fields valid")
                        else:
                            if result['missing_fields']:
                                st.warning("Missing Fields:")
                                for field in result['missing_fields']:
                                    st.markdown(f"- {field}")
                            
                            if result['invalid_fields']:
                                st.warning("Invalid Fields:")
                                for field in result['invalid_fields']:
                                    st.markdown(f"- {field}")
                        
                        # Display extracted data
                        st.write("**Extracted Data:**")
                        display_extracted_data(result['extracted_data'])

def main():
    """
    Main application function.
    """
    # Title and description
    st.title("Document Classification and Data Extraction System")
    st.markdown("""
    This application enables you to:
    - Train a document classification model
    - Classify uploaded documents
    - Extract structured data from documents
    - Configure extraction patterns
    - Process documents in batch
    """)
    
    # Create sidebar
    st.sidebar.title("Navigation")
    
    # Navigation options
    pages = {
        "Classify Document": classify_document_ui,
        "Train Model": train_model_ui,
        "Extraction Configuration": extraction_config_ui,
        "Batch Processing": batch_processing_ui
    }
    
    # Select page
    page = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Display selected page
    pages[page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 Document Classification & Extraction System")

if __name__ == "__main__":
    main()