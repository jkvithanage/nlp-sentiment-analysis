from bs4 import BeautifulSoup
import os
import glob

# Base path where the files are stored
base_path = "./data/"

def load_data():
    """
    Load all review data from .review files in the data folder.
    
    Returns:
        list: Array of tuples (review_text, sentiment) where:
            - review_text: string containing the review text
            - sentiment: string containing sentiment label ('positive'/'negative')
    """
    all_reviews = []
    
    print("Loading review data...")
    
    # Find all .review files recursively in the data directory
    review_files = glob.glob(os.path.join(base_path, "**", "*.review"), recursive=True)
    
    for file_path in review_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = _parse_reviews(file.read())
            
            relative_path = os.path.relpath(file_path, base_path)
            print(f"  {relative_path}: {len(data)} reviews")
            
            for text, sentiment in data:
                all_reviews.append((text, sentiment))
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    return all_reviews

def _parse_reviews(xml_content):
    # Wrap content in a root element to handle multiple review elements
    wrapped_content = f'<reviews>{xml_content}</reviews>'
    soup = BeautifulSoup(wrapped_content, 'xml')
    reviews = soup.find_all('review')
    
    data = []
    for review in reviews:
        # Check if review_text exists and is not empty
        review_text_element = review.find('review_text')
        rating_element = review.find('rating')
        
        if review_text_element is None or rating_element is None:
            continue  # Skip reviews missing required elements
            
        text = review_text_element.get_text(strip=True)
        if not text:  # Skip reviews with empty text
            continue
            
        try:
            rating = float(rating_element.get_text(strip=True))
        except (ValueError, TypeError):
            continue  # Skip reviews with invalid ratings
            
        sentiment = 'positive' if rating >= 4.0 else 'negative'
        data.append((text, sentiment))
    
    return data
