import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from preprocess import load_and_split

def run_baseline():
    print("Dataset load aur split ho raha hai...")
    train, val, test = load_and_split("../data/raw/Roman Urdu DataSet.csv")
    
    # TF-IDF (Char n-grams) + Logistic Regression Pipeline
    # Char-wb n-grams (2 to 5) Roman Urdu typos/spelling variations captures karti hain
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced", # Handles class imbalance
            random_state=42
        ))
    ])
    
    print("Baseline model train ho raha hai...")
    pipeline.fit(train["text"], train["label"])
    
    print("\n================ TEST SET PERFORMANCE ================")
    test_preds = pipeline.predict(test["text"])
    print(classification_report(test["label"], test_preds))
    
    macro_f1 = f1_score(test["label"], test_preds, average="macro")
    print(f"--> Final Baseline Test Macro-F1 Score: {macro_f1:.4f}")
    
    # Save baseline model
    joblib.dump(pipeline, "../models/tfidf_baseline.joblib")
    print("\nModel saved to: ml/models/tfidf_baseline.joblib")

if __name__ == "__main__":
    run_baseline()