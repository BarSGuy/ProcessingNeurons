import random
import time
import os
import json

def generate_unique_10_digit_number():
    # Seed the random number generator with the current time
    random.seed(time.time())
    number = random.randint(1000000000, 9999999999)
    return number


def save_dict_to_path(data, path):
    """
    Save a dictionary to a JSON file at the given path. Create directories if they don't exist.
    
    Parameters:
    data (dict): The dictionary to save.
    path (str): The file path where the dictionary should be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the dictionary as a JSON file
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Dictionary of best metrics are saved to {path}")

def load_dict_from_path(path):
    """
    Load a dictionary from a JSON file at the given path.
    
    Parameters:
    path (str): The file path from which the dictionary should be loaded.
    
    Returns:
    dict: The dictionary loaded from the JSON file.
    """
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data
