import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score
from preprocess import load_and_split

# 1. Configuration & Labels Setup
MODEL_NAME = "xlm-roberta-base"
LABELS = ["Negative", "Neutral", "Positive"]
l2i = {label: idx for idx, label in enumerate(LABELS)}

def prepare_data():
    print("Dataset load aur split ho raha hai...")
    train, val, test = load_and_split("../data/raw/Roman Urdu DataSet.csv")
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def to_ds(df):
        ds = Dataset.from_dict({
            "text": df["text"].tolist(), 
            "label": df["label"].map(l2i).tolist()
        })
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=128), 
            batched=True
        )

    print("Data tokenize ho raha hai...")
    train_ds = to_ds(train)
    val_ds = to_ds(val)
    test_ds = to_ds(test)
    
    return train_ds, val_ds, test_ds, tok

def metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds)
    }

def main():
    train_ds, val_ds, test_ds, tok = prepare_data()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3, 
        id2label=dict(enumerate(LABELS)), 
        label2id=l2i
    )
    
    args = TrainingArguments(
        output_dir="../models/xlmr",
        learning_rate=2e-5,
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=True,  # Colab GPU acceleration
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics
    )
    
    print("Training start ho rahi hai...")
    trainer.train()
    
    print("\n================ TEST SET EVALUATION ================")
    test_results = trainer.evaluate(test_ds)
    print(test_results)
    
    # Local temporary save inside Colab
    trainer.save_model("../models/xlmr-final")
    tok.save_pretrained("../models/xlmr-final")
    
    # Push directly to Hugging Face Hub
    # UPDATE THIS WITH YOUR ACTUAL HUGGING FACE USERNAME!
    HF_REPO = "FatimaMansoorAhmed/urdusense-xlmr" 
    
    print(f"\nModel ko Hugging Face Hub par push kar rahe hain: {HF_REPO}")
    trainer.model.push_to_hub(HF_REPO, private=True)
    tok.push_to_hub(HF_REPO, private=True)
    print("Push Complete! Model Hub par save ho gaya hai.")

if __name__ == "__main__":
    main()