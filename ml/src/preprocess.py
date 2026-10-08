import re
import pandas as pd
from sklearn.model_selection import train_test_split

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
REPEAT = re.compile(r"(.)\1{2,}")      # acchaaaa -> accha (keeps 2)
SPACES = re.compile(r"\s+")

def clean(text: str) -> str:
    text = str(text).lower()
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = REPEAT.sub(r"\1\1", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)  # Keep Latin + Urdu script
    return SPACES.sub(" ", text).strip()

def load_and_split(path: str, seed: int = 42):
    # Encoding & error handling added for clean CSV reading
    df = pd.read_csv(
        path, 
        header=None, 
        usecols=[0, 1], 
        names=["text", "label"], 
        encoding="utf-8", 
        on_bad_lines="skip"
    )
    
    # Strip whitespace from string labels (e.g. "Positive " -> "Positive")
    df["label"] = df["label"].astype(str).str.strip()
    
    df = df.dropna()
    df = df[df["label"].isin(["Positive", "Neutral", "Negative"])]  # Drops junk rows
    
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text")                          # Deduplicate before splitting

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=seed)
    
    return train, val, test

if __name__ == "__main__":
    # Test script locally
    train, val, test = load_and_split("../data/raw/Roman Urdu DataSet.csv")
    print(f"Train size: {len(train)} | Val size: {len(val)} | Test size: {len(test)}")
    print("\nTrain Label Distribution:\n", train["label"].value_counts())
    