import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

HF_REPO = "FatimaFM26/urdusense-xlmr"
LOCAL_MODEL_DIR = "../models/xlmr-final"

def load_model():
    # Download model from Hugging Face Hub if not already local
    if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
        print("Hugging Face Hub se model download ho raha hai...")
        snapshot_download(repo_id=HF_REPO, local_dir=LOCAL_MODEL_DIR)
        print("Download complete!")
    else:
        print("Local cached model load ho raha hai...")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

def main():
    pipe = load_model()
    
    test_samples = [
        "yeh boht achi film thi mujhe boht maza aya",
        "bakwas service thi bilkul time waste hua",
        "kal weather kaisa rahega islamabad mein"
    ]
    
    print("\n================ MODEL PREDICTION TEST ================")
    for sample in test_samples:
        res = pipe(sample)[0]
        print(f"Text: '{sample}'")
        print(f"Predicted Label: {res['label']} | Confidence: {res['score']:.4f}\n")

if __name__ == "__main__":
    main()