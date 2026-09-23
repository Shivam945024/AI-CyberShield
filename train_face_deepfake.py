```python
import os
import json
import torch

from torchvision import datasets, transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "google/vit-base-patch16-224"

TRAIN_DIR = "data/face_deepfake/train"
VAL_DIR = "data/face_deepfake/validation"

OUTPUT_DIR = "models/face_deepfake"

EPOCHS = 2
BATCH_SIZE = 4
IMAGE_SIZE = 224


# =========================================================
# CREATE FOLDERS
# =========================================================

folders = [
    f"{TRAIN_DIR}/real",
    f"{TRAIN_DIR}/deepfake",
    f"{VAL_DIR}/real",
    f"{VAL_DIR}/deepfake",
    OUTPUT_DIR,
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# =========================================================
# CHECK DATASET
# =========================================================

train_real = len(os.listdir(f"{TRAIN_DIR}/real"))
train_fake = len(os.listdir(f"{TRAIN_DIR}/deepfake"))

val_real = len(os.listdir(f"{VAL_DIR}/real"))
val_fake = len(os.listdir(f"{VAL_DIR}/deepfake"))

print("\n====================================")
print("AI-CyberShield Deepfake Training")
print("====================================")

print(f"\nTrain REAL      : {train_real}")
print(f"Train DEEPFAKE  : {train_fake}")
print(f"Validation REAL : {val_real}")
print(f"Validation FAKE : {val_fake}")


if train_real == 0 or train_fake == 0:
    print("\n❌ Training images missing!")

    print("\nPut images here:")

    print("data/face_deepfake/train/real/")
    print("data/face_deepfake/train/deepfake/")

    print("\nThen run:")
    print("python train_face_deepfake.py")

    exit()


if val_real == 0 or val_fake == 0:
    print("\n❌ Validation images missing!")

    print("\nPut images here:")

    print("data/face_deepfake/validation/real/")
    print("data/face_deepfake/validation/deepfake/")

    exit()


# =========================================================
# IMAGE PROCESSOR
# =========================================================

print("\nLoading AI model...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

mean = processor.image_mean
std = processor.image_std


# =========================================================
# IMAGE TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# =========================================================
# DATASET
# =========================================================

class DeepfakeDataset(torch.utils.data.Dataset):

    def __init__(self, folder, transform):

        self.dataset = datasets.ImageFolder(
            folder
        )

        self.transform = transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        image, label = self.dataset[index]

        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "labels": label
        }


train_dataset = DeepfakeDataset(
    TRAIN_DIR,
    train_transform
)

val_dataset = DeepfakeDataset(
    VAL_DIR,
    val_transform
)


# =========================================================
# CLASS MAPPING
# =========================================================

class_to_idx = train_dataset.dataset.class_to_idx

print("\nClass mapping:")
print(class_to_idx)

# ImageFolder normally gives:
# deepfake = 0
# real = 1

id2label = {
    value: key.upper()
    for key, value in class_to_idx.items()
}

label2id = {
    key.upper(): value
    for key, value in class_to_idx.items()
}

print("\nModel labels:")
print(id2label)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading ViT model...")

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


# =========================================================
# TRAINING SETTINGS
# =========================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    evaluation_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_steps=5,

    save_total_limit=1,

    report_to="none",

    fp16=torch.cuda.is_available(),
)


# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


# =========================================================
# START TRAINING
# =========================================================

print("\n====================================")
print("🚀 TRAINING STARTED")
print("====================================")

if torch.cuda.is_available():
    print("GPU detected")

else:
    print("CPU detected")
    print("Training may take some time.")


trainer.train()


# =========================================================
# SAVE MODEL
# =========================================================

print("\n====================================")
print("💾 SAVING MODEL")
print("====================================")

trainer.save_model(OUTPUT_DIR)

processor.save_pretrained(OUTPUT_DIR)


# =========================================================
# SAVE CLASS INFO
# =========================================================

with open(
    f"{OUTPUT_DIR}/classes.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "classes": id2label,
            "label2id": label2id
        },
        file,
        indent=4
    )


# =========================================================
# FINISHED
# =========================================================

print("\n====================================")
print("✅ TRAINING COMPLETED")
print("====================================")

print("\nModel location:")

print(
    os.path.abspath(OUTPUT_DIR)
)

print("\nFiles created:")

for file in os.listdir(OUTPUT_DIR):

    print("  ✅", file)

print("\nNow run:")

print("streamlit run app.py")

print("\nYour Deepfake AI Model should show:")
print("🤖 Deepfake AI Model: READY")
```
