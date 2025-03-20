"""
Unit tests for the data extraction module.
"""

import os
import unittest
import tempfile
import json
import yaml
from unittest.mock import patch, MagicMock
from models.extractor import DataExtractor

class TestDataExtractor(unittest.TestCase):
    """Test cases for data extractor."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Sample texts for different document types
        self.invoice_text = """
        INVOICE
        
        Invoice Number: INV-12345
        Invoice Date: 01/15/2023
        
        Bill To:
        ABC Corporation
        123 Main Street, New York, NY 10001
        Contact: John Smith
        
        Description                  Quantity    Unit Price    Amount
        ----------------------------------------------------------------
        Web Development Services     40          $75.00        $3,000.00
        UX/UI Design                 20          $85.00        $1,700.00
        
        Subtotal: $4,700.00
        Tax (8.0%): $376.00
        Total: $5,076.00
        
        Payment Terms: Net 30 days
        """
        
        self.resume_text = """
        JANE DOE
        
        Email: jane.doe@example.com
        Phone: (555) 123-4567
        Address: 456 Park Avenue, San Francisco, CA 94107
        
        PROFESSIONAL SUMMARY
        
        Experienced software developer with over 8 years of expertise in web development, 
        database design, and cloud computing.
        
        SKILLS
        
        Python, JavaScript, React, Node.js, SQL, MongoDB, AWS, Docker, Git, Agile Development
        
        WORK EXPERIENCE
        
        Senior Software Engineer | Tech Solutions Inc.
        January 2018 - Present
        
        - Developed scalable web applications using React and Node.js
        - Led a team of 5 developers on multiple projects
        - Implemented CI/CD pipelines using Jenkins and GitHub Actions
        
        Software Developer | Digital Innovations LLC
        March 2015 - December 2017
        
        - Built RESTful APIs for mobile applications
        - Optimized database queries to improve application performance
        - Collaborated with UX designers to implement responsive designs
        
        EDUCATION
        
        Bachelor of Science in Computer Science
        University of California, Berkeley, Graduated 2014
        
        CERTIFICATIONS
        
        AWS Certified Solutions Architect
        Certified Scrum Master (CSM)
        """
        
        self.contract_text = """
        SERVICE AGREEMENT
        
        This Service Agreement (the "Agreement") is made effective as of January 10, 2023, by and between:
        
        Global Tech Solutions, Inc. ("Client"), a company organized and existing under the laws of the State of California, 
        with its principal place of business located at 123 Tech Avenue, San Francisco, CA 94105
        
        and
        
        Professional Services Group, LLC ("Contractor"), a company organized and existing under the laws of the State of New York, 
        with its principal place of business located at 456 Business Lane, New York, NY 10001
        
        The parties agree as follows:
        
        1. SERVICES
        
        Contractor shall provide software development services to Client.
        
        2. PAYMENT
        
        Client will pay Contractor $150.00 per hour for the Services.
        Payment shall be made within 30 days of receiving Contractor's invoice.
        
        3. TERM
        
        This Agreement shall begin on January 10, 2023 and continue for a period of 1 year.
        
        4. CONFIDENTIALITY
        
        During the term of this Agreement, and for a period of two (2) years after its termination, 
        Contractor shall not disclose to any third party any confidential information of Client without Client's prior written consent.
        
        5. TERMINATION
        
        Either party may terminate this Agreement with 30 days written notice to the other party.
        
        SIGNATURES
        
        _____________________________
        For Global Tech Solutions, Inc.
        
        Name: John Executive
        Title: CEO
        Date: January 10, 2023
        
        _____________________________
        For Professional Services Group, LLC
        
        Name: Sarah Manager
        Title: Managing Director
        Date: January 10, 2023
        """
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        extractor = DataExtractor()
        
        # Check if default configuration is loaded
        self.assertIn('invoice', extractor.config)
        self.assertIn('resume', extractor.config)
        self.assertIn('contract', extractor.config)
        
        # Check structure of default config for invoice
        self.assertIn('patterns', extractor.config['invoice'])
        self.assertIn('entities', extractor.config['invoice'])
        
        # Check specific patterns and entities exist
        self.assertIn('invoice_number', extractor.config['invoice']['patterns'])
        self.assertIn('date', extractor.config['invoice']['patterns'])
        self.assertIn('total_amount', extractor.config['invoice']['patterns'])
        
        # Check that entities include common types
        invoice_entities = extractor.config['invoice']['entities']
        self.assertTrue(any(entity in invoice_entities for entity in ['ORG', 'DATE', 'MONEY']))
    
    def test_load_config_json(self):
        """Test loading configuration from a JSON file."""
        # Create a test JSON config file
        config = {
            'test_type': {
                'patterns': {
                    'field1': 'pattern1',
                    'field2': 'pattern2'
                },
                'entities': ['ENTITY1', 'ENTITY2']
            }
        }
        
        json_path = os.path.join(self.temp_dir.name, 'test_config.json')
        with open(json_path, 'w') as f:
            json.dump(config, f)
        
        # Load the config
        extractor = DataExtractor(json_path)
        
        # Check if config was loaded correctly
        self.assertIn('test_type', extractor.config)
        self.assertEqual(extractor.config['test_type']['patterns']['field1'], 'pattern1')
        self.assertEqual(extractor.config['test_type']['patterns']['field2'], 'pattern2')
        self.assertEqual(extractor.config['test_type']['entities'], ['ENTITY1', 'ENTITY2'])
    
    def test_load_config_yaml(self):
        """Test loading configuration from a YAML file."""
        # Create a test YAML config file
        config = {
            'test_type': {
                'patterns': {
                    'field1': 'pattern1',
                    'field2': 'pattern2'
                },
                'entities': ['ENTITY1', 'ENTITY2']
            }
        }
        
        yaml_path = os.path.join(self.temp_dir.name, 'test_config.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        
        # Load the config
        extractor = DataExtractor(yaml_path)
        
        # Check if config was loaded correctly
        self.assertIn('test_type', extractor.config)
        self.assertEqual(extractor.config['test_type']['patterns']['field1'], 'pattern1')
        self.assertEqual(extractor.config['test_type']['patterns']['field2'], 'pattern2')
        self.assertEqual(extractor.config['test_type']['entities'], ['ENTITY1', 'ENTITY2'])
    
    def test_extract_data_invoice(self):
        """Test extracting data from an invoice document."""
        extractor = DataExtractor()
        result = extractor.extract_data(self.invoice_text, 'invoice')
        
        # Check if expected fields were extracted
        self.assertEqual(result['document_type'], 'invoice')
        self.assertEqual(result['invoice_number'], 'INV-12345')
        self.assertEqual(result['date'], '01/15/2023')
        self.assertEqual(result['total_amount'], '5,076.00')
        
        # Optional: Check for extraction of line items if implemented
        if 'line_items' in result:
            self.assertIsInstance(result['line_items'], list)
            if result['line_items']:  # If any items were extracted
                self.assertIn('description', result['line_items'][0])
                self.assertIn('quantity', result['line_items'][0])
                self.assertIn('unit_price', result['line_items'][0])
                self.assertIn('amount', result['line_items'][0])
    
    def test_extract_data_resume(self):
        """Test extracting data from a resume document."""
        extractor = DataExtractor()
        result = extractor.extract_data(self.resume_text, 'resume')
        
        # Check if expected fields were extracted
        self.assertEqual(result['document_type'], 'resume')
        self.assertEqual(result['name'], 'JANE DOE')
        self.assertEqual(result['email'], 'jane.doe@example.com')
        self.assertEqual(result['phone'], '(555) 123-4567')
        
        # Check for skills extraction
        if 'skills' in result:
            self.assertIn('Python', result['skills'])
        
        # Check for skills list if implemented
        if 'skills_list' in result:
            self.assertIsInstance(result['skills_list'], list)
            self.assertTrue(any('Python' in skill for skill in result['skills_list']))
        
        # Check for work experience sections if implemented
        if 'experience_sections' in result:
            self.assertIsInstance(result['experience_sections'], list)
            if result['experience_sections']:
                self.assertIn('company', result['experience_sections'][0])
                self.assertIn('position', result['experience_sections'][0])
                self.assertIn('start_date', result['experience_sections'][0])
                self.assertIn('end_date', result['experience_sections'][0])
    
    def test_extract_data_contract(self):
        """Test extracting data from a contract document."""
        extractor = DataExtractor()
        result = extractor.extract_data(self.contract_text, 'contract')
        
        # Check if expected fields were extracted
        self.assertEqual(result['document_type'], 'contract')
        self.assertTrue('parties' in result or ('entities_ORG' in result and len(result['entities_ORG']) >= 2))
        self.assertEqual(result['effective_date'], 'January 10, 2023')
        self.assertTrue('term' in result and '1 year' in result['term'])
        self.assertTrue('payment' in result and '$150.00 per hour' in result['payment'])
        
        # Check for signatories if implemented
        if 'signatories' in result:
            self.assertIsInstance(result['signatories'], list)
            self.assertTrue(any('John' in signatory for signatory in result['signatories']) or 
                           any('Sarah' in signatory for signatory in result['signatories']))
        
        # Check for dates if implemented
        if 'dates' in result:
            self.assertIsInstance(result['dates'], list)
            self.assertTrue(any('January 10, 2023' in date for date in result['dates']))
    
    def test_save_config(self):
        """Test saving configuration to a file."""
        extractor = DataExtractor()
        
        # Add a test pattern
        extractor.add_extraction_pattern('test_type', 'test_field', 'test_pattern')
        
        # Save to JSON
        json_path = os.path.join(self.temp_dir.name, 'saved_config.json')
        extractor.save_config(json_path)
        
        # Check if file was created
        self.assertTrue(os.path.exists(json_path))
        
        # Load the saved config and check if it contains our test pattern
        with open(json_path, 'r') as f:
            saved_config = json.load(f)
        
        self.assertIn('test_type', saved_config)
        self.assertIn('patterns', saved_config['test_type'])
        self.assertIn('test_field', saved_config['test_type']['patterns'])
        self.assertEqual(saved_config['test_type']['patterns']['test_field'], 'test_pattern')
        
        # Save to YAML
        yaml_path = os.path.join(self.temp_dir.name, 'saved_config.yaml')
        extractor.save_config(yaml_path)
        
        # Check if file was created
        self.assertTrue(os.path.exists(yaml_path))
        
        # Load the saved config and check if it contains our test pattern
        with open(yaml_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertIn('test_type', saved_config)
        self.assertIn('patterns', saved_config['test_type'])
        self.assertIn('test_field', saved_config['test_type']['patterns'])
        self.assertEqual(saved_config['test_type']['patterns']['test_field'], 'test_pattern')
    
    def test_add_extraction_pattern(self):
        """Test adding an extraction pattern."""
        extractor = DataExtractor()
        
        # Add a new pattern to an existing document type
        extractor.add_extraction_pattern('invoice', 'purchase_order', r'(?i)P\.?O\.?\s*(?:#|number|num|no)?[:\s]*([A-Z0-9\-]+)')
        
        # Check if pattern was added
        self.assertIn('purchase_order', extractor.config['invoice']['patterns'])
        self.assertEqual(
            extractor.config['invoice']['patterns']['purchase_order'],
            r'(?i)P\.?O\.?\s*(?:#|number|num|no)?[:\s]*([A-Z0-9\-]+)'
        )
        
        # Add a pattern to a new document type
        extractor.add_extraction_pattern('new_type', 'new_field', 'new_pattern')
        
        # Check if document type and pattern were added
        self.assertIn('new_type', extractor.config)
        self.assertIn('patterns', extractor.config['new_type'])
        self.assertIn('new_field', extractor.config['new_type']['patterns'])
        self.assertEqual(extractor.config['new_type']['patterns']['new_field'], 'new_pattern')
    
    def test_add_entity_type(self):
        """Test adding an entity type."""
        extractor = DataExtractor()
        
        # Add a new entity type to an existing document type
        extractor.add_entity_type('invoice', 'PRODUCT')
        
        # Check if entity type was added
        self.assertIn('PRODUCT', extractor.config['invoice']['entities'])
        
        # Add an entity type to a new document type
        extractor.add_entity_type('new_type', 'NEW_ENTITY')
        
        # Check if document type and entity type were added
        self.assertIn('new_type', extractor.config)
        self.assertIn('entities', extractor.config['new_type'])
        self.assertIn('NEW_ENTITY', extractor.config['new_type']['entities'])
    
    def test_extract_data_unsupported_type(self):
        """Test extracting data with an unsupported document type."""
        extractor = DataExtractor()
        result = extractor.extract_data("Some test content", "unsupported_type")
        
        # Should return an empty dict except for document_type
        self.assertEqual(result, {'document_type': 'unsupported_type'})
    
    @patch('models.extractor.extract_named_entities')
    def test_extract_named_entities_integration(self, mock_extract_entities):
        """Test integration with named entity extraction."""
        # Mock the entity extraction function
        mock_extract_entities.return_value = {
            'ORG': ['ABC Corporation', 'XYZ Inc.'],
            'PERSON': ['John Smith'],
            'DATE': ['January 15, 2023'],
            'MONEY': ['$5,076.00']
        }
        
        extractor = DataExtractor()
        result = extractor.extract_data(self.invoice_text, 'invoice')
        
        # Check if entities were extracted and included in result
        self.assertIn('entities_ORG', result)
        self.assertIn('entities_PERSON', result)
        self.assertIn('entities_DATE', result)
        self.assertIn('entities_MONEY', result)
        
        # Check entity values
        self.assertEqual(result['entities_ORG'], ['ABC Corporation', 'XYZ Inc.'])
        self.assertEqual(result['entities_PERSON'], ['John Smith'])
        self.assertEqual(result['entities_DATE'], ['January 15, 2023'])
        self.assertEqual(result['entities_MONEY'], ['$5,076.00'])
    
    def test_load_config_invalid_format(self):
        """Test loading configuration with invalid format."""
        # Create a file with invalid extension
        invalid_path = os.path.join(self.temp_dir.name, 'invalid_config.txt')
        with open(invalid_path, 'w') as f:
            f.write("This is not a valid config file")
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            DataExtractor(invalid_path)

if __name__ == '__main__':
    unittest.main()