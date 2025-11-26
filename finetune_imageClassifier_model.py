import os
import torch
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import json

def transform_images(examples):
    examples['pixel_values'] = [
        processor(images=image.convert("RGB"), return_tensors="pt")['pixel_values'].squeeze(0)
        for image in examples['image']
    ]
    return examples


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}


def collate_fn(examples):
    """Custom collate function for the trainer"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
    


MODEL_NAME = "google/vit-base-patch16-224-in21k"
DATASET_NAME = "mvkvc/artifact-100k"
LOCAL_SAVING_PATH = "./finetuned_artifact_model"
ds = load_dataset(DATASET_NAME)

id2label = {0: "ai", 1: "real"}
label2id = {"ai": 0, 'real': 1}

ds_train = ds['train']
ds_test = ds['test']


processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels = 2, id2label=id2label, label2id=label2id).to(torch.device("cuda" if torch.cuda.is_available() else 'cpu'))


print("Processing datasets...")
processed_train = ds_train.map(transform_images, batched=True,  batch_size=32,remove_columns=['image'])
processed_test = ds_test.map(transform_images, batched=True, batch_size=32, remove_columns=['image'])


processed_train.set_format(type='torch', columns=['pixel_values', 'label'])
processed_test.set_format(type='torch', columns=['pixel_values', 'label'])


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-05,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
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
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_test,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()
print("Training completed.")

print("Evaluating model...")
eval_metrics = trainer.evaluate()
print("Evaluation metrics:", eval_metrics)


os.makedirs("./results", exist_ok=True)
metrics_path = os.path.join("./results", "eval_results_image.json")
with open(metrics_path, "w") as f:
    json.dump(eval_metrics, f, indent=4)

print(f"Evaluation metrics saved to {metrics_path}")

# Save model and processor
trainer.save_model(LOCAL_SAVING_PATH)
processor.save_pretrained(LOCAL_SAVING_PATH)

print(f"Model and processor saved locally to {LOCAL_SAVING_PATH}")