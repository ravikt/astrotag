# File: file_handler.py
import json
import os
from typing import Dict


class FileHandler:
    @staticmethod
    def save_to_json(data: Dict, filename: str, directory: str = "output"):
        """
        Save dictionary to a JSON file with proper formatting.
        
        Args:
            data: Dictionary to save
            filename: Name of the output file
            directory: Directory to save the file in (default: "output")
        """
        # Create output directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Construct full file path
        filepath = os.path.join(directory, filename)
        
        # Save the data
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath

    @staticmethod
    def load_from_json(filename: str, directory: str = "output") -> Dict:
        """
        Load dictionary from a JSON file.
        
        Args:
            filename: Name of the file to load
            directory: Directory containing the file (default: "output")
            
        Returns:
            Dictionary containing the loaded data
        """
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'r') as f:
            return json.load(f)