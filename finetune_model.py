import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
import json


def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding= 'max_length', max_length = 128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

MODEL_NAME = "vinai/bertweet-base"
DATASET_NAME = "roupenminassian/twitter-misinformation"
HUB_MODEL_ID = "IoanRoume/twitter-misinformation-bertweet" 
LOCAL_SAVING_PATH = "./finetuned_bertweet_model"

ds = load_dataset("roupenminassian/twitter-misinformation")

id2label = {0: "Legitimate", 1: "Misinformation"}
label2id = {"Legitimate": 0, "Misinformation": 1}


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model.config.id2label = id2label
model.config.label2id = label2id


tokenized_train = ds['train'].map(tokenize_function, batched=True)
tokenized_test = ds['test'].map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",           
    eval_strategy="epoch",
    save_strategy="epoch",     
    learning_rate=2e-05,              
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,              
    weight_decay=0.01,               
    save_total_limit=1,              
    load_best_model_at_end=True,     
    logging_dir="./logs",            
    logging_steps=100,               
    fp16=torch.cuda.is_available(),
    lr_scheduler_type="linear",
    optim="adamw_torch",
    metric_for_best_model="f1",
    report_to="none"
)




trainer = Trainer(
    model=model,
    args= training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()
print("Training completed.")

print("Evaluating model...")
eval_metrics = trainer.evaluate()
print("Evaluation metrics:", eval_metrics)

# Ensure results directory exists
os.makedirs("./results", exist_ok=True)
metrics_path = os.path.join("./results", "eval_results.json")
with open(metrics_path, "w") as f:
    json.dump(eval_metrics, f, indent=4)

print(f"Evaluation metrics saved to {metrics_path}")

trainer.save_model(LOCAL_SAVING_PATH)
tokenizer.save_pretrained(LOCAL_SAVING_PATH) 

print(f"Model saved locally to {LOCAL_SAVING_PATH}")