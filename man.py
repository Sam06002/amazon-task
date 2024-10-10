import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    pandas.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data

def generate_synthetic_review(prompt, max_length=200):
    """
    Generate a synthetic review using the BLOOM model via Hugging Face API.
    
    Args:
    prompt (str): The initial text to base the generated review on.
    max_length (int): Maximum number of tokens to generate.
    
    Returns:
    str: Generated review text, or None if generation failed.
    """
    # API endpoint for the BLOOM model
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
    
    # Set up authentication header with API key
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    
    # Define parameters for text generation
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": 0.7,  # Controls randomness: lower is more deterministic
            "top_p": 0.9,  # Controls diversity of generated text
        }
    }
    
    # Send POST request to the API
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def generate_synthetic_dataset(original_reviews, num_samples=100):
    """
    Generate a dataset of synthetic reviews based on original reviews.
    
    Args:
    original_reviews (list): List of original review texts.
    num_samples (int): Number of synthetic reviews to generate.
    
    Returns:
    list: List of generated synthetic reviews.
    """
    synthetic_reviews = []
    for i in range(num_samples):
        # Randomly select a prompt from original reviews
        prompt = np.random.choice(original_reviews)
        if prompt:
            # Generate a synthetic review based on the prompt
            synthetic_review = generate_synthetic_review(prompt)
            if synthetic_review:
                synthetic_reviews.append(synthetic_review)
                print(f"Generated {i+1}/{num_samples} synthetic reviews")
            else:
                print(f"Failed to generate review {i+1}/{num_samples}")
        else:
            print("Empty prompt, skipping...")
    return synthetic_reviews

def save_synthetic_dataset(synthetic_reviews, output_file):
    """
    Save the generated synthetic reviews to a CSV file.
    
    Args:
    synthetic_reviews (list): List of generated synthetic reviews.
    output_file (str): Path to save the output CSV file.
    """
    df_synthetic = pd.DataFrame(synthetic_reviews, columns=['Synthetic_Review'])
    df_synthetic.to_csv(output_file, index=False)
    print(f"Synthetic dataset saved to {output_file}")

def main():
    """
    Main function to orchestrate the synthetic review generation process.
    """
    # Path to the original dataset
    original_dataset_path = '/Users/shivamsrivastava/Downloads/Projects/amazon-task-main/reviews_supplements.csv'
    
    # Load the original dataset
    data = load_dataset(original_dataset_path)
    
    # Extract review texts, dropping any null values
    original_reviews = data['text'].dropna().tolist()
    
    # Set the number of synthetic samples to generate
    num_synthetic_samples = 100
    
    # Generate synthetic reviews
    synthetic_reviews = generate_synthetic_dataset(original_reviews, num_synthetic_samples)
    
    # Define output file path
    output_file = 'synthetic_reviews.csv'
    
    # Save the generated synthetic reviews
    save_synthetic_dataset(synthetic_reviews, output_file)

if __name__ == "__main__":
    main()