import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')  # Tokenizer
nltk.download('stopwords')  # Stop words for filtering
nltk.download('wordnet')  # WordNet lemmatizer

# Preprocessing Functions

def remove_urls(text, replacement_text=""):
    """
    Removes URLs from the given text.
    
    Args:
    text: Input string potentially containing URLs.
    replacement_text: Text to replace the URLs with (default is empty string).
    
    Returns:
    Text with URLs removed.
    """
    pattern = re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub(replacement_text, text)

def remove_twitter_handles(text, replacement_text=""):
    """
    Removes Twitter handles from the text (e.g., '@username').
    """
    pattern = re.compile(r"@[\w]+")
    return pattern.sub(replacement_text, text)

def remove_twitter_rt(text, replacement_text=""):
    """
    Removes retweet ('RT') markers from the text.
    """
    pattern = re.compile(r"^RT|\s+RT\s+")
    return pattern.sub(replacement_text, text)

def remove_alphanumerics(text, replacement_text=" "):
    """
    Removes non-alphanumeric characters, except for single quotes.
    """
    pattern = re.compile(r"[^A-Za-z0-9']+")
    return pattern.sub(replacement_text, text)

def remove_multiple_whitespaces(text, replacement_text=" "):
    """
    Replaces multiple consecutive whitespaces with a single space.
    """
    pattern = re.compile(r"\s{2,}")
    return pattern.sub(replacement_text, text)

def decode_html_character_references(text):
    """
    Decodes HTML character references (e.g., '&amp;' to '&').
    """
    import html
    return html.unescape(text)

def tokenize(doc):
    """
    Tokenizes the text into individual words.
    """
    return word_tokenize(doc)

def remove_stopwords(doc):
    """
    Removes English stop words from the tokenized text.
    
    Args:
    doc: List of tokens.
    
    Returns:
    Filtered list of tokens without stop words.
    """
    stops = set(stopwords.words("english"))
    stops.add("rt")  # Adding 'rt' (retweet marker) to stop words
    return [token for token in doc if token not in stops]

def decontracted(phrase):
    """
    Expands common English contractions in the given text.
    
    Args:
    phrase: Input string with contractions.
    
    Returns:
    String with contractions expanded (e.g., "can't" to "cannot").
    """
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"ain\'t", "are not", phrase)
    phrase = re.sub(r"shan\'t", "shall not", phrase)
    phrase = re.sub(r"ma\'am", "maam", phrase)
    phrase = re.sub(r"y\'all", "you all", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and Preprocess Data

def load_and_preprocess_data(path):
    """
    Loads and preprocesses the dataset for training models.
    
    Steps:
    1. Load the data from the specified CSV file.
    2. Drop irrelevant columns.
    3. Map class labels to binary values.
    4. Clean tweets using text preprocessing functions.
    5. Tokenize and remove stop words.
    
    Args:
    path: Path to the CSV file containing the data.
    
    Returns:
    Preprocessed Pandas DataFrame with clean and tokenized text.
    """
    # Load the dataset into a Pandas DataFrame
    df = pd.read_csv(path)
    
    # Preprocessing steps
    df = (df
        .drop(columns=["count", "hate_speech_count", "offensive_language_count", "neither_count"])  # Drop irrelevant columns
        .assign(class_=df["class"].map({0: 1, 1: 1, 2: 0}))  # Map class labels: 0,1 to 1 (positive), 2 to 0 (negative)
        .rename(columns={"class_": "class"})  # Rename column for clarity
        .assign(tweet_clean=lambda df_: (
            df_["tweet"]
            .apply(decode_html_character_references)  # Decode HTML entities
            .apply(remove_twitter_handles)  # Remove Twitter handles
            .apply(remove_twitter_rt)  # Remove retweet markers
            .apply(remove_urls)  # Remove URLs
            .apply(remove_alphanumerics)  # Remove non-alphanumeric characters
            .apply(remove_multiple_whitespaces)  # Replace multiple spaces with a single space
            .str.strip()))  # Remove leading/trailing whitespaces
        .assign(tweet_preprocessed=lambda df_: (
            df_["tweet_clean"]
            .str.lower()  # Convert to lowercase
            .apply(lambda doc: [decontracted(word) for word in doc.split(" ")])  # Expand contractions
            .apply(lambda doc: [lemmatizer.lemmatize(word) for word in doc])  # Lemmatize each word
            .apply(lambda doc: " ".join(doc))  # Reconstruct text
            .apply(word_tokenize)  # Tokenize text
            .apply(remove_stopwords)))  # Remove stop words
    )
    
    return df