"""
Document preprocessing utilities for text extraction and normalization.
"""

import os
import re
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import logging

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If the model is not available, download it
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfplumber and pytesseract as fallback.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    try:
        # First try using pdfplumber (works for text-based PDFs)
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # If pdfplumber extracted text successfully, return it
            if text.strip():
                return text
            
        # If pdfplumber didn't get text (likely a scanned PDF), use OCR
        logger.info(f"Using OCR for {pdf_path} as pdfplumber couldn't extract text")
        images = convert_from_path(pdf_path)
        text = ""
        for i, image in enumerate(images):
            text += pytesseract.image_to_string(image) + "\n"
        return text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def extract_text_from_docx(docx_path):
    """
    Extract text from a DOCX file.
    
    Args:
        docx_path (str): Path to the DOCX file
        
    Returns:
        str: Extracted text
    """
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {docx_path}: {str(e)}")
        return ""

def extract_text_from_image(image_path):
    """
    Extract text from an image file using OCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image {image_path}: {str(e)}")
        return ""

def extract_text(file_path):
    """
    Extract text from a file based on its extension.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Extracted text
    """
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        return extract_text_from_image(file_path)
    elif ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            return ""
    else:
        logger.warning(f"Unsupported file format: {ext}")
        return ""

def preprocess_text(text):
    """
    Preprocess text by removing special characters, lowercasing, and removing stopwords.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(filtered_text)

def extract_named_entities(text):
    """
    Extract named entities from text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of named entities grouped by entity type
    """
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    
    return entities