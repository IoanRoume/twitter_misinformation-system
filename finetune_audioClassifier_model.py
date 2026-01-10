from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, Audio
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import json
import os


def compute_metrics(eval_results):
    logits,labels = eval_results
    outputs = np.argmax(logits, axis=-1)
    acc =accuracy_score(labels,outputs)
    f1 = f1_score(labels,outputs, average='weighted')
    return {"accuracy": acc, "f1": f1}


def transform_audios(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000 * 10,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    return inputs

MODEL_NAME = "facebook/wav2vec2-base"
DATASET_NAME = "Hemg/Deepfakeaudio"
LOCAL_SAVING_PATH = "./finetuned_audio_model"


ds = load_dataset(DATASET_NAME)

ds = ds['train'].train_test_split(test_size = 0.2, seed = 42)
ds_train = ds['train']
ds_test = ds['test']
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

id2label={
    0:"fake",
    1:"real"
}

label2id={
    "fake":0,
    "real":1
}

model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME,num_labels = 2, id2label=id2label, label2id=label2id)



ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000))

processed_train = ds_train.map(
    transform_audios,
    batched=True,
    batch_size=32,
    remove_columns=[col for col in ds_train.column_names if col != 'label']
)

processed_test = ds_test.map(
    transform_audios,
    batched=True,
    batch_size=32,
    remove_columns=[col for col in ds_test.column_names if col != 'label']
)


print("Preprocessing complete!")


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-05,
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(), 
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    metric_for_best_model="f1",
    report_to="none",
    push_to_hub=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_test,
    processing_class=feature_extractor, 
    compute_metrics=compute_metrics
)


print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

trainer.train()

print("\n" + "="*50)
print("Training completed!")
print("="*50 + "\n")


print("Evaluating model...")
eval_metrics = trainer.evaluate()

print("\nEvaluation metrics:")
for key, value in eval_metrics.items():
    print(f"  {key}: {value:.4f}")


os.makedirs("./results", exist_ok=True)
metrics_path = os.path.join("./results", "eval_results_audio.json")

with open(metrics_path, "w") as f:
    json.dump(eval_metrics, f, indent=4)

print(f"\nEvaluation metrics saved to {metrics_path}")


print(f"\nSaving model to {LOCAL_SAVING_PATH}...")
trainer.save_model(LOCAL_SAVING_PATH)
feature_extractor.save_pretrained(LOCAL_SAVING_PATH)

print(f"Model and feature extractor saved to {LOCAL_SAVING_PATH}")