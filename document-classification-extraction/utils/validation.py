"""
Validation utilities for document processing and data extraction.
"""

import os
import logging
from utils.preprocessing import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentValidator:
    """
    Validate documents and extracted data.
    """
    
    @staticmethod
    def validate_file(file_path):
        """
        Validate that a file exists and is a supported document type.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if the file is valid, False otherwise
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
        
        # Check file extension
        _, ext = os.path.splitext(file_path.lower())
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        if ext not in supported_extensions:
            logger.error(f"Unsupported file format: {ext}")
            return False
        
        # Check if file is readable and contains text
        try:
            text = extract_text(file_path)
            if not text:
                logger.warning(f"No text could be extracted from file: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return False
        
        return True
    
    @staticmethod
    def validate_extracted_data(data, document_type):
        """
        Validate extracted data for completeness and correctness.
        
        Args:
            data (dict): Extracted data
            document_type (str): Type of document
            
        Returns:
            dict: Validation results
        """
        validation_result = {
            'is_valid': True,
            'missing_fields': [],
            'invalid_fields': []
        }
        
        # Check for required fields based on document type
        required_fields = DocumentValidator._get_required_fields(document_type)
        
        for field in required_fields:
            if field not in data or not data[field]:
                validation_result['missing_fields'].append(field)
                validation_result['is_valid'] = False
        
        # Validate specific fields
        if document_type == 'invoice':
            DocumentValidator._validate_invoice_fields(data, validation_result)
        elif document_type == 'resume':
            DocumentValidator._validate_resume_fields(data, validation_result)
        elif document_type == 'contract':
            DocumentValidator._validate_contract_fields(data, validation_result)
        
        return validation_result
    
    @staticmethod
    def _get_required_fields(document_type):
        """
        Get required fields for a document type.
        
        Args:
            document_type (str): Type of document
            
        Returns:
            list: Required fields
        """
        if document_type == 'invoice':
            return ['invoice_number', 'date', 'total_amount']
        elif document_type == 'resume':
            return ['name', 'email']
        elif document_type == 'contract':
            return ['parties', 'effective_date']
        else:
            return []
    
    @staticmethod
    def _validate_invoice_fields(data, validation_result):
        """
        Validate invoice-specific fields.
        
        Args:
            data (dict): Extracted data
            validation_result (dict): Validation results to update
        """
        # Validate invoice number format
        if 'invoice_number' in data and data['invoice_number']:
            if not DocumentValidator._is_valid_invoice_number(data['invoice_number']):
                validation_result['invalid_fields'].append('invoice_number')
                validation_result['is_valid'] = False
        
        # Validate date format
        if 'date' in data and data['date']:
            if not DocumentValidator._is_valid_date(data['date']):
                validation_result['invalid_fields'].append('date')
                validation_result['is_valid'] = False
        
        # Validate amount format
        if 'total_amount' in data and data['total_amount']:
            if not DocumentValidator._is_valid_amount(data['total_amount']):
                validation_result['invalid_fields'].append('total_amount')
                validation_result['is_valid'] = False
    
    @staticmethod
    def _validate_resume_fields(data, validation_result):
        """
        Validate resume-specific fields.
        
        Args:
            data (dict): Extracted data
            validation_result (dict): Validation results to update
        """
        # Validate email format
        if 'email' in data and data['email']:
            if not DocumentValidator._is_valid_email(data['email']):
                validation_result['invalid_fields'].append('email')
                validation_result['is_valid'] = False
        
        # Validate phone format
        if 'phone' in data and data['phone']:
            if not DocumentValidator._is_valid_phone(data['phone']):
                validation_result['invalid_fields'].append('phone')
                validation_result['is_valid'] = False
    
    @staticmethod
    def _validate_contract_fields(data, validation_result):
        """
        Validate contract-specific fields.
        
        Args:
            data (dict): Extracted data
            validation_result (dict): Validation results to update
        """
        # Validate date format
        if 'effective_date' in data and data['effective_date']:
            if not DocumentValidator._is_valid_date(data['effective_date']):
                validation_result['invalid_fields'].append('effective_date')
                validation_result['is_valid'] = False
    
    @staticmethod
    def _is_valid_invoice_number(invoice_number):
        """
        Validate invoice number format.
        
        Args:
            invoice_number (str): Invoice number
            
        Returns:
            bool: True if the invoice number is valid, False otherwise
        """
        # Simple validation: at least one letter or digit
        return bool(invoice_number and any(c.isalnum() for c in invoice_number))
    
    @staticmethod
    def _is_valid_date(date):
        """
        Validate date format.
        
        Args:
            date (str): Date string
            
        Returns:
            bool: True if the date is valid, False otherwise
        """
        # Simple validation for common date formats
        import re
        patterns = [
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',  # DD/MM/YYYY, MM/DD/YYYY
            r'\w+\s+\d{1,2},?\s+\d{4}'  # Month DD, YYYY
        ]
        return any(re.fullmatch(pattern, date) for pattern in patterns)
    
    @staticmethod
    def _is_valid_amount(amount):
        """
        Validate amount format.
        
        Args:
            amount (str): Amount string
            
        Returns:
            bool: True if the amount is valid, False otherwise
        """
        # Simple validation for numeric amount, possibly with commas and decimal point
        import re
        return bool(re.fullmatch(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?', amount))
    
    @staticmethod
    def _is_valid_email(email):
        """
        Validate email format.
        
        Args:
            email (str): Email string
            
        Returns:
            bool: True if the email is valid, False otherwise
        """
        import re
        return bool(re.fullmatch(r'[\w\.-]+@[\w\.-]+\.\w+', email))
    
    @staticmethod
    def _is_valid_phone(phone):
        """
        Validate phone number format.
        
        Args:
            phone (str): Phone number string
            
        Returns:
            bool: True if the phone number is valid, False otherwise
        """
        import re
        patterns = [
            r'\+\d{1,2}\s*\(\d{3}\)\s*\d{3}[-\s]?\d{4}',  # +1 (123) 456-7890
            r'\(\d{3}\)\s*\d{3}[-\s]?\d{4}',  # (123) 456-7890
            r'\d{3}[-\s]?\d{3}[-\s]?\d{4}'  # 123-456-7890
        ]
        return any(re.fullmatch(pattern, phone) for pattern in patterns)