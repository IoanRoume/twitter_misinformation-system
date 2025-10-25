import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding= 'max_length', max_length = 128)

ds = load_dataset("roupenminassian/twitter-misinformation")
model_name = "vinai/bertweet-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


tokenized_train = ds['train'].map(tokenize_function, batched=True)
tokenized_test = ds['test'].map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",           
    eval_strategy="epoch",
    save_strategy="epoch",     
    learning_rate=2e-05,              
    per_device_train_batch_size=64,  # With 140 gb GPU, can go up to 64
    per_device_eval_batch_size=64,
    num_train_epochs=3,              
    weight_decay=0.01,               
    save_total_limit=1,              
    load_best_model_at_end=True,     
    logging_dir="./logs",            
    logging_steps=100,               
    fp16=True,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    metric_for_best_model="f1"
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicitions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predicitions)
    f1 = f1_score(labels, predicitions, average='weighted')
    return {"accuracy": acc, "f1": f1}

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

saving_path = "./finetuned_bertweet_model"
trainer.save_model(saving_path)
print(f"Model saved to {saving_path}")

