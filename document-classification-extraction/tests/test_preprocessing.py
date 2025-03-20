"""
Unit tests for the preprocessing utilities.
"""

import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from utils.preprocessing import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_image,
    extract_text,
    preprocess_text,
    extract_named_entities
)

class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing utilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def create_temp_file(self, content, extension):
        """Create a temporary file with specified content and extension."""
        file_path = os.path.join(self.temp_dir.name, f"test_file{extension}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_preprocess_text(self):
        """Test text preprocessing functionality."""
        # Test case with mixed case, punctuation, numbers, and stopwords
        input_text = "This is a SAMPLE text with 123 numbers and punctuation! The, a, an are stopwords."
        expected_output = "sample text numbers punctuation stopwords"
        processed_text = preprocess_text(input_text)
        self.assertEqual(processed_text, expected_output)
        
        # Test empty string
        self.assertEqual(preprocess_text(""), "")
        
        # Test string with only stopwords and punctuation
        input_text = "The, a, an, and, of, to, in, with, for!"
        expected_output = ""
        processed_text = preprocess_text(input_text)
        self.assertEqual(processed_text, expected_output)
    
    @patch('utils.preprocessing.extract_text_from_pdf')
    @patch('utils.preprocessing.extract_text_from_docx')
    @patch('utils.preprocessing.extract_text_from_image')
    def test_extract_text(self, mock_extract_image, mock_extract_docx, mock_extract_pdf):
        """Test extract_text function with different file types."""
        # Set up mocks
        mock_extract_pdf.return_value = "PDF content"
        mock_extract_docx.return_value = "DOCX content"
        mock_extract_image.return_value = "Image content"
        
        # Test PDF extraction
        pdf_path = os.path.join(self.temp_dir.name, "test.pdf")
        self.assertEqual(extract_text(pdf_path), "PDF content")
        mock_extract_pdf.assert_called_once_with(pdf_path)
        
        # Test DOCX extraction
        docx_path = os.path.join(self.temp_dir.name, "test.docx")
        self.assertEqual(extract_text(docx_path), "DOCX content")
        mock_extract_docx.assert_called_once_with(docx_path)
        
        # Test image extraction
        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            image_path = os.path.join(self.temp_dir.name, f"test{ext}")
            mock_extract_image.reset_mock()
            self.assertEqual(extract_text(image_path), "Image content")
            mock_extract_image.assert_called_once_with(image_path)
        
        # Test text file extraction
        content = "This is a test text file."
        txt_path = self.create_temp_file(content, ".txt")
        self.assertEqual(extract_text(txt_path), content)
        
        # Test unsupported format
        unsupported_path = os.path.join(self.temp_dir.name, "test.xyz")
        self.assertEqual(extract_text(unsupported_path), "")
    
    @patch('utils.preprocessing.pdfplumber.open')
    @patch('utils.preprocessing.convert_from_path')
    @patch('utils.preprocessing.pytesseract.image_to_string')
    def test_extract_text_from_pdf(self, mock_image_to_string, mock_convert_from_path, mock_pdfplumber_open):
        """Test PDF text extraction."""
        # Mock pdfplumber for text-based PDF
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted text from PDF"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_path = os.path.join(self.temp_dir.name, "test.pdf")
        
        # Test extraction from text-based PDF
        result = extract_text_from_pdf(pdf_path)
        self.assertEqual(result, "Extracted text from PDF\n")
        mock_pdfplumber_open.assert_called_once_with(pdf_path)
        mock_convert_from_path.assert_not_called()
        
        # Reset mocks for scanned PDF test
        mock_pdfplumber_open.reset_mock()
        mock_page.extract_text.return_value = ""  # No text extractable
        
        # Mock OCR for scanned PDF
        mock_image = MagicMock()
        mock_convert_from_path.return_value = [mock_image]
        mock_image_to_string.return_value = "OCR extracted text"
        
        # Test extraction from scanned PDF
        result = extract_text_from_pdf(pdf_path)
        self.assertEqual(result, "OCR extracted text\n")
        mock_convert_from_path.assert_called_once_with(pdf_path)
        mock_image_to_string.assert_called_once_with(mock_image)
    
    @patch('utils.preprocessing.Document')
    def test_extract_text_from_docx(self, mock_document):
        """Test DOCX text extraction."""
        # Set up mock document
        mock_doc = MagicMock()
        mock_document.return_value = mock_doc
        mock_doc.paragraphs = [
            MagicMock(text="Paragraph 1"),
            MagicMock(text="Paragraph 2"),
            MagicMock(text="Paragraph 3")
        ]
        
        docx_path = os.path.join(self.temp_dir.name, "test.docx")
        result = extract_text_from_docx(docx_path)
        
        self.assertEqual(result, "Paragraph 1\nParagraph 2\nParagraph 3")
        mock_document.assert_called_once_with(docx_path)
    
    @patch('utils.preprocessing.Image.open')
    @patch('utils.preprocessing.pytesseract.image_to_string')
    def test_extract_text_from_image(self, mock_image_to_string, mock_image_open):
        """Test image text extraction."""
        # Set up mock image
        mock_img = MagicMock()
        mock_image_open.return_value = mock_img
        mock_image_to_string.return_value = "Text extracted from image"
        
        image_path = os.path.join(self.temp_dir.name, "test.jpg")
        result = extract_text_from_image(image_path)
        
        self.assertEqual(result, "Text extracted from image")
        mock_image_open.assert_called_once_with(image_path)
        mock_image_to_string.assert_called_once_with(mock_img)
    
    @patch('utils.preprocessing.nlp')
    def test_extract_named_entities(self, mock_nlp):
        """Test named entity extraction."""
        # Set up mock entities
        mock_ent1 = MagicMock(text="John Doe", label_="PERSON")
        mock_ent2 = MagicMock(text="New York", label_="GPE")
        mock_ent3 = MagicMock(text="Google", label_="ORG")
        mock_ent4 = MagicMock(text="$1000", label_="MONEY")
        mock_ent5 = MagicMock(text="January 1, 2023", label_="DATE")
        
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent1, mock_ent2, mock_ent3, mock_ent4, mock_ent5]
        mock_nlp.return_value = mock_doc
        
        text = "John Doe from New York works at Google and earns $1000 as of January 1, 2023."
        entities = extract_named_entities(text)
        
        expected_entities = {
            'PERSON': ['John Doe'],
            'GPE': ['New York'],
            'ORG': ['Google'],
            'MONEY': ['$1000'],
            'DATE': ['January 1, 2023']
        }
        
        self.assertEqual(entities, expected_entities)
        mock_nlp.assert_called_once_with(text)

if __name__ == '__main__':
    unittest.main()