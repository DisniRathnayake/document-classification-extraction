"""
Data extraction module for extracting structured information from classified documents.
"""

import re
import os
import json
import yaml
import logging
import spacy
from utils.preprocessing import extract_named_entities

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If the model is not available, download it
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class DataExtractor:
    """
    Extract structured information from documents based on their category.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the data extractor.
        
        Args:
            config_path (str): Path to extraction configuration file
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)
        else:
            # Default extraction configurations
            self.config = {
                'invoice': {
                    'patterns': {
                        'invoice_number': r'(?i)invoice\s*(?:#|number|num|no)?[:\s]*([A-Z0-9\-]+)',
                        'date': r'(?i)(?:invoice\s*)?date[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
                        'total_amount': r'(?i)(?:total|amount|sum)[:\s]*[$€£]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                        'vendor': r'(?i)(?:vendor|seller|from)[:\s]*([A-Za-z0-9\s\.,]+?)(?:\s*\n|\s*$|\s*,)',
                        'client': r'(?i)(?:client|customer|bill to|to)[:\s]*([A-Za-z0-9\s\.,]+?)(?:\s*\n|\s*$|\s*,)'
                    },
                    'entities': ['ORG', 'DATE', 'MONEY', 'PERSON']
                },
                'resume': {
                    'patterns': {
                        'name': r'(?i)^([A-Za-z\s\.-]+)$',
                        'email': r'[\w\.-]+@[\w\.-]+\.\w+',
                        'phone': r'(?:\+\d{1,2}\s*)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}',
                        'education': r'(?i)(?:education|degree|university|college)[:\s]*([A-Za-z0-9\s\.,]+?)(?:\s*\n|\s*$|\s*,)',
                        'skills': r'(?i)(?:skills|expertise|proficiencies)[:\s]*([A-Za-z0-9\s\.,\+\#]+?)(?:\s*\n|\s*$|\s*,)',
                        'experience': r'(?i)(?:experience|work experience|employment)[:\s]*([A-Za-z0-9\s\.,]+?)(?:\s*\n|\s*$|\s*,)'
                    },
                    'entities': ['PERSON', 'ORG', 'DATE', 'GPE', 'SKILL']
                },
                'contract': {
                    'patterns': {
                        'parties': r'(?i)(?:between|party)[:\s]*([A-Za-z0-9\s\.,]+?)[:\s]*and[:\s]*([A-Za-z0-9\s\.,]+?)(?:\s*\n|\s*$|\s*,)',
                        'effective_date': r'(?i)(?:effective date|commencement date|start date)[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
                        'term': r'(?i)(?:term|duration|period)[:\s]*([A-Za-z0-9\s\.,]+?)(?:\s*\n|\s*$|\s*,)',
                        'payment': r'(?i)(?:payment|compensation|fee)[:\s]*([A-Za-z0-9\s\.,\$]+?)(?:\s*\n|\s*$|\s*,)',
                        'termination': r'(?i)(?:termination|cancellation)[:\s]*([A-Za-z0-9\s\.,]+?)(?:\s*\n|\s*$|\s*,)'
                    },
                    'entities': ['ORG', 'DATE', 'MONEY', 'PERSON', 'GPE']
                }
            }
    
    def load_config(self, config_path):
        """
        Load extraction configuration from a file (JSON or YAML).
        
        Args:
            config_path (str): Path to the configuration file
        """
        try:
            _, ext = os.path.splitext(config_path.lower())
            
            if ext == '.json':
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            elif ext in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
            
            logger.info(f"Loaded extraction configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def extract_data(self, text, document_type):
        """
        Extract structured information from document text based on its type.
        
        Args:
            text (str): Document text
            document_type (str): Type of document (e.g., 'invoice', 'resume')
            
        Returns:
            dict: Extracted structured information
        """
        if document_type not in self.config:
            logger.warning(f"No extraction configuration found for document type: {document_type}")
            return {'document_type': document_type}
        
        # Initialize result with document type
        result = {'document_type': document_type}
        
        try:
            # Extract data using regular expression patterns
            patterns = self.config[document_type].get('patterns', {})
            for field, pattern in patterns.items():
                try:
                    match = re.search(pattern, text)
                    if match:
                        # Check if the pattern has a capturing group
                        if match.lastindex is not None:
                            result[field] = match.group(1).strip()
                        else:
                            # If no capturing group, use the whole match
                            result[field] = match.group(0).strip()
                except Exception as e:
                    logger.warning(f"Error extracting {field} with pattern {pattern}: {str(e)}")
            
            # Extract named entities
            if 'entities' in self.config[document_type]:
                try:
                    entities = extract_named_entities(text)
                    for entity_type in self.config[document_type]['entities']:
                        if entity_type in entities:
                            result[f'entities_{entity_type}'] = entities[entity_type]
                except Exception as e:
                    logger.warning(f"Error extracting named entities: {str(e)}")
            
            # Document type-specific extraction
            try:
                if document_type == 'invoice':
                    self._extract_invoice_specific(text, result)
                elif document_type == 'resume':
                    self._extract_resume_specific(text, result)
                elif document_type == 'contract':
                    self._extract_contract_specific(text, result)
            except Exception as e:
                logger.warning(f"Error in document type-specific extraction: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error extracting data from {document_type}: {str(e)}")
        
        return result
    
    def _extract_invoice_specific(self, text, result):
        """
        Extract invoice-specific information.
        
        Args:
            text (str): Document text
            result (dict): Current extraction results to update
        """
        # Extract line items if present
        try:
            line_items = []
            line_item_pattern = r'(?i)(\d+|[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:x\s+)?(\d+(?:\.\d+)?)\s+(?:@\s+)?[$€£]?(\d+(?:\.\d+)?)\s+[$€£]?(\d+(?:\.\d+)?)'
            
            for match in re.finditer(line_item_pattern, text):
                item = {
                    'description': match.group(1).strip(),
                    'quantity': match.group(2).strip(),
                    'unit_price': match.group(3).strip(),
                    'amount': match.group(4).strip()
                }
                line_items.append(item)
            
            if line_items:
                result['line_items'] = line_items
        except Exception as e:
            logger.warning(f"Error extracting line items: {str(e)}")
    
    def _extract_resume_specific(self, text, result):
        """
        Extract resume-specific information.
        
        Args:
            text (str): Document text
            result (dict): Current extraction results to update
        """
        # Extract skills list
        try:
            if 'skills' in result:
                skills_text = result['skills']
                skills = [skill.strip() for skill in re.split(r'[,;]', skills_text) if skill.strip()]
                result['skills_list'] = skills
        except Exception as e:
            logger.warning(f"Error extracting skills list: {str(e)}")
        
        # Extract work experience sections
        try:
            experience_sections = []
            exp_pattern = r'(?i)(?:^|\n)([A-Za-z0-9\s\.,]+?)\s*[-–|]\s*([A-Za-z0-9\s\.,]+?)\s*[-–|]\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{4})\s*[-–|to]+\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{4}|present|current)'
            
            for match in re.finditer(exp_pattern, text):
                exp = {
                    'company': match.group(1).strip(),
                    'position': match.group(2).strip(),
                    'start_date': match.group(3).strip(),
                    'end_date': match.group(4).strip()
                }
                experience_sections.append(exp)
            
            if experience_sections:
                result['experience_sections'] = experience_sections
        except Exception as e:
            logger.warning(f"Error extracting experience sections: {str(e)}")
    
    def _extract_contract_specific(self, text, result):
        """
        Extract contract-specific information.
        
        Args:
            text (str): Document text
            result (dict): Current extraction results to update
        """
        # Extract signatories
        try:
            signature_pattern = r'(?i)(?:signed|signature)[:\s]*([A-Za-z\s\.]+)'
            signatures = []
            
            for match in re.finditer(signature_pattern, text):
                signatures.append(match.group(1).strip())
            
            if signatures:
                result['signatories'] = signatures
        except Exception as e:
            logger.warning(f"Error extracting signatories: {str(e)}")
        
        # Extract dates from contract
        try:
            date_pattern = r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b'
            dates = []
            
            for match in re.finditer(date_pattern, text):
                date = match.group(1).strip()
                if date not in dates:
                    dates.append(date)
            
            if dates:
                result['dates'] = dates
        except Exception as e:
            logger.warning(f"Error extracting dates: {str(e)}")
    
    def save_config(self, config_path):
        """
        Save the current extraction configuration to a file.
        
        Args:
            config_path (str): Path to save the configuration
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            _, ext = os.path.splitext(config_path.lower())
            
            if ext == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif ext in ['.yml', '.yaml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
            
            logger.info(f"Saved extraction configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def add_extraction_pattern(self, document_type, field, pattern):
        """
        Add or update an extraction pattern for a document type.
        
        Args:
            document_type (str): Type of document
            field (str): Field name to extract
            pattern (str): Regular expression pattern
        """
        # Create document type config if it doesn't exist
        if document_type not in self.config:
            self.config[document_type] = {'patterns': {}, 'entities': []}
        
        # Create patterns dict if it doesn't exist
        if 'patterns' not in self.config[document_type]:
            self.config[document_type]['patterns'] = {}
        
        # Add or update the pattern
        self.config[document_type]['patterns'][field] = pattern
        logger.info(f"Added extraction pattern for {document_type}.{field}")
    
    def add_entity_type(self, document_type, entity_type):
        """
        Add an entity type to extract for a document type.
        
        Args:
            document_type (str): Type of document
            entity_type (str): Entity type to extract (e.g., 'PERSON', 'ORG')
        """
        # Create document type config if it doesn't exist
        if document_type not in self.config:
            self.config[document_type] = {'patterns': {}, 'entities': []}
        
        # Create entities list if it doesn't exist
        if 'entities' not in self.config[document_type]:
            self.config[document_type]['entities'] = []
        
        # Add entity type if not already present