from bs4 import BeautifulSoup
import os
import glob
import csv

# Base path where the files are stored
base_path = "./data/"

def load_data(data_dir=base_path) -> list[tuple[str, float]]:
    """
    Load all review data from .review files in the data folder.

    Returns:
        list: list of tuples (review_text, sentiment) where:
            - review_text: string containing the review text
            - rating: rating value out of 5
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

        data.append((text, rating))

    return data


if __name__ == "__main__":
    data = load_data()
    with open('raw_data.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

    print("Raw data saved to raw_data.csv")
