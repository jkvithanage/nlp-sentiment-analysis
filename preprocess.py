from data_loader import load_data
from typing import List
import re

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
except Exception:
    nltk = None


def verify_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def remove_stopwords(tokens: List[str], stop_set: set) -> List[str]:
    """Remove stopwords from tokens."""
    return [t for t in tokens if t not in stop_set]


def lemmatize_tokens(tokens: List[str], lemmatizer: WordNetLemmatizer) -> List[str]:
    """Lemmatize tokens using NLTK's WordNetLemmatizer."""
    return [lemmatizer.lemmatize(tok) for tok in tokens]


def preprocess_word(word: str) -> str:
    """Word level preprocessing"""
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r"(.)\1+", r"\1\1", word)
    # Remove hyphens and apostrophes
    word = re.sub(r"[-']", "", word)
    return word


def is_valid_word(word: str) -> bool:
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def preprocess_reviews(raw_data: List[tuple[str, float]]) -> List[tuple[str, int]]:
    """Preprocess reviews"""
    if nltk is None:
        raise SystemExit("NLTK is required. Please `pip install nltk`.")
    verify_nltk_resources()
    sw = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    processed_data = []

    print('Preprocessing reviews...')
    for i, (review, rating) in enumerate(raw_data):
        if i % 1000 == 0:
            print(f"Processing review {i+1}/{len(raw_data)}")

        # Convert rating to sentiment (1 for positive, 0 for negative)
        # Ratings 1-2 are negative, 4-5 are positive, 3 is neutral (excluded)
        if rating <= 2:
            sentiment = 0  # Negative
        elif rating >= 4:
            sentiment = 1  # Positive
        else:
            continue  # Skip neutral ratings (3)

        # Convert to lower case and remove whitespace
        review = review.lower().strip()
        # Remove undesired underscore and slashes
        review = review.replace("_", " ").replace("/", "")

        # Remove Numbers
        review = re.sub(r"\b[0-9]+\b\s*", "", review)

        # Remove Hyperlinks
        review = re.sub(r"https?://\S+", "", review)

        # Remove the <a> tags
        review = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", review)

        # Remove the HTML tags but keep their contents
        review = re.sub(r"<.*?>", " ", review)

        # Remove the alphanumerics like a389794njfhj because they don't add any value
        review = re.sub(r'\w*\d\w*', '', review)

        # Remove undesired punctuation
        processed_tokens = RegexpTokenizer(r'\w+').tokenize(review)

        # Process word by word
        processed_tokens = [preprocess_word(word) for word in processed_tokens if is_valid_word(word)]

        # Remove stopwords
        processed_tokens = remove_stopwords(processed_tokens, sw)

        # Lemmatize
        processed_tokens = lemmatize_tokens(processed_tokens, lemmatizer)

        # Join tokens back into text
        processed_review = " ".join(processed_tokens)
        # Skip if text ended up empty after cleaning
        if not processed_review:
            continue
        processed_data.append((processed_review, sentiment))

    return processed_data


def preprocess_text_for_inference(text: str) -> str:
    """
    Preprocess a single review text using the same steps
    """
    if text is None:
        return ""
    if nltk is None:
        raise SystemExit("NLTK is required. Please `pip install nltk`.")
    verify_nltk_resources()

    sw = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Text-level cleaning (mirror training flow)
    text = text.lower().strip()
    text = text.replace("_", " ").replace("/", "")
    text = re.sub(r"\b[0-9]+\b\s*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)

    tokens = RegexpTokenizer(r"\w+").tokenize(text)
    tokens = [preprocess_word(w) for w in tokens if is_valid_word(w)]
    tokens = remove_stopwords(tokens, sw)
    tokens = lemmatize_tokens(tokens, lemmatizer)
    return " ".join(tokens)


# Optional alias referenced by the UI if present
def clean_for_vectorizer(text: str) -> str:
    return preprocess_text_for_inference(text)


if __name__ == "__main__":
    # Load the raw data
    print("Loading data...")
    raw_data = load_data()
    print(f"Loaded {len(raw_data)} reviews")

    processed_data = preprocess_reviews(raw_data)

    import csv
    with open('preprocessed.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'label'])  # Header
        writer.writerows(processed_data)

    print(f"Preprocessing complete! Saved {len(processed_data)} preprocessed reviews to preprocessed.csv")
    print(f"Positive reviews: {sum(1 for _, s in processed_data if s == 1)}")
    print(f"Negative reviews: {sum(1 for _, s in processed_data if s == 0)}")
