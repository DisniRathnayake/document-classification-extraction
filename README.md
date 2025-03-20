# Document Classification and Data Extraction System

A comprehensive system for classifying documents and extracting structured information with a user-friendly Streamlit interface.

## Features

- **Document Classification:** Classify uploaded documents into predefined categories (invoices, resumes, contracts)
- **Data Extraction:** Extract structured information from documents based on their classification
- **User Interface:** Intuitive Streamlit UI for document processing and visualization
- **Batch Processing:** Process multiple documents in a single batch
- **Configurable Extraction:** Customize extraction patterns for different document types


## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for extracting text from images and scanned PDFs)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/DisniRathnayake/document-classification-extraction.git
   cd document-classification-extraction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Training a Classification Model

1. Navigate to the "Train Model" page in the sidebar
2. Upload example documents for each category (invoices, resumes, contracts in data/training) 
3. Configure the model parameters
4. Click "Train Model"

### Classifying and Extracting Data

1. Navigate to the "Classify Document" page in the sidebar
2. Upload a document (data/sample_documents)
3. View the classification results and extracted information

### Batch Processing

1. Navigate to the "Batch Processing" page in the sidebar
2. Upload multiple documents
3. Click "Process Batch"
4. Download the results or view detailed information for each document

### Configuring Extraction Patterns

1. Navigate to the "Extraction Configuration" page in the sidebar
2. View and edit extraction patterns for each document type
3. Add new patterns or entity types
4. Save the configuration

## Architecture

### Components

- **Preprocessing Module:** Handles text extraction and normalization
- **Classification Model:** Categorizes documents using machine learning
- **Data Extraction Module:** Extracts structured information using regex patterns and NER
- **Validation Module:** Validates extracted data for completeness and correctness
- **Streamlit UI:** Provides a user-friendly interface for all functionality

### File Structure

```
doc_classifier/
├── README.md                       # Project documentation
├── requirements.txt                # Dependencies
├── app.py                          # Main Streamlit application
├── data/
│   ├── training/                   # Training data for document classifier
│   │   ├── invoices/               # Sample invoice documents
│   │   ├── resumes/                # Sample resume documents
│   │   └── contracts/              # Sample contract documents
│   └── sample_documents/           # Sample documents for testing
├── models/
│   ├── classifier.py               # Document classification model
│   └── extractor.py                # Data extraction module
├── utils/
│   ├── preprocessing.py            # Document preprocessing utilities
│   ├── feature_extraction.py       # Feature extraction for classification
│   └── validation.py               # Validation utilities
└── tests/                          # Unit tests
    ├── test_classifier.py
    ├── test_extractor.py
    └── test_preprocessing.py
```

## Configuration

### Classification Models

The system supports three classification algorithms:
- **Support Vector Machine (SVM):** Default option with good performance
- **Logistic Regression:** Alternative for smaller datasets
- **Random Forest:** Option for more complex classification tasks

### Extraction Patterns

Extraction patterns are configured using regular expressions for each document type:

- **Invoices:** Extracts invoice numbers, dates, amounts, vendor/client information, and line items
- **Resumes:** Extracts personal information, education, skills, and work experience
- **Contracts:** Extracts parties, effective dates, terms, payment details, and termination clauses

## Sample Data

The repository includes sample documents for training and testing:
- Sample invoices
- Sample resumes
- Sample contracts

## Testing

Run the tests to verify the functionality:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
