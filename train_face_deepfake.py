```python
import os
import json
import random

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "google/vit-base-patch16-224"

CSV_FILE = "data/face_deepfake/face_deepfake_dataset.csv"

OUTPUT_DIR = "models/face_deepfake"

IMAGE_SIZE = 224

EPOCHS = 2

BATCH_SIZE = 4

LEARNING_RATE = 2e-5

SEED = 42


# =========================================================
# SEED
# =========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# CHECK CSV
# =========================================================

print("\n======================================")
print("AI-CyberShield Deepfake Trainer")
print("======================================")

print("\nCSV file:")

print(CSV_FILE)


if not os.path.exists(CSV_FILE):

    raise FileNotFoundError(
        f"\nCSV file not found:\n{CSV_FILE}"
    )


# =========================================================
# READ CSV
# =========================================================

df = pd.read_csv(CSV_FILE)


print("\nCSV loaded successfully.")

print("\nColumns:")

print(df.columns.tolist())


# =========================================================
# CHECK COLUMNS
# =========================================================

required_columns = [
    "image_path",
    "label"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nMissing column: {column}"
        )


# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=["image_path", "label"]
)


df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# =========================================================
# CHECK LABELS
# =========================================================

allowed_labels = {
    "real",
    "deepfake"
}


invalid_labels = set(
    df["label"].unique()
) - allowed_labels


if invalid_labels:

    raise ValueError(
        f"\nInvalid labels found: {invalid_labels}\n"
        "Allowed labels: real, deepfake"
    )


# =========================================================
# CHECK IMAGE FILES
# =========================================================

print("\nChecking images...")

missing_images = []

valid_rows = []


for index, row in df.iterrows():

    image_path = str(
        row["image_path"]
    )

    if os.path.exists(image_path):

        valid_rows.append(row)

    else:

        missing_images.append(
            image_path
        )


if missing_images:

    print(
        f"\n⚠️ Missing images: {len(missing_images)}"
    )

    for path in missing_images[:10]:

        print(
            "  Missing:",
            path
        )


df = pd.DataFrame(valid_rows)


if len(df) == 0:

    raise ValueError(
        "\nNo valid image files found."
    )


# =========================================================
# DATASET SUMMARY
# =========================================================

print("\n======================================")
print("DATASET SUMMARY")
print("======================================")

print(
    "\nTotal valid images:",
    len(df)
)


print(
    "\nREAL:",
    len(df[df["label"] == "real"])
)


print(
    "DEEPFAKE:",
    len(df[df["label"] == "deepfake"])
)


if len(df[df["label"] == "real"]) == 0:

    raise ValueError(
        "No REAL images found."
    )


if len(df[df["label"] == "deepfake"]) == 0:

    raise ValueError(
        "No DEEPFAKE images found."
    )


# =========================================================
# LABEL MAPPING
# =========================================================

label2id = {
    "REAL": 0,
    "DEEPFAKE": 1
}


id2label = {
    0: "REAL",
    1: "DEEPFAKE"
}


# =========================================================
# PROCESSOR
# =========================================================

print("\nLoading image processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)


mean = processor.image_mean

std = processor.image_std


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# =========================================================
# SPLIT DATA
# =========================================================

df = df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


split_index = int(
    len(df) * 0.8
)


train_df = df.iloc[
    :split_index
].reset_index(drop=True)


val_df = df.iloc[
    split_index:
].reset_index(drop=True)


print("\nTraining images:", len(train_df))

print("Validation images:", len(val_df))


# =========================================================
# DATASET CLASS
# =========================================================

class DeepfakeDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform
    ):

        self.df = dataframe

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]

        label_name = row["label"]


        # Load image

        image = Image.open(
            image_path
        ).convert("RGB")


        # Transform

        if self.transform:

            image = self.transform(
                image
            )


        # Label

        label = label2id[
            label_name.upper()
        ]


        return {
            "pixel_values": image,
            "labels": label
        }


# =========================================================
# CREATE DATASETS
# =========================================================

train_dataset = DeepfakeDataset(
    train_df,
    train_transform
)


val_dataset = DeepfakeDataset(
    val_df,
    val_transform
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading ViT model...")

model = AutoModelForImageClassification.from_pretrained(

    MODEL_NAME,

    num_labels=2,

    id2label=id2label,

    label2id=label2id,

    ignore_mismatched_sizes=True
)


# =========================================================
# TRAINING ARGUMENTS
# =========================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_steps=5,

    save_total_limit=1,

    report_to="none",

    fp16=torch.cuda.is_available()
)


# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset
)


# =========================================================
# TRAIN
# =========================================================

print("\n======================================")

print("🚀 TRAINING STARTED")

print("======================================")


if torch.cuda.is_available():

    print("\nGPU detected.")

else:

    print(
        "\nCPU detected."
    )

    print(
        "Training may take some time."
    )


trainer.train()


# =========================================================
# EVALUATE
# =========================================================

print("\n======================================")

print("EVALUATING MODEL")

print("======================================")


results = trainer.evaluate()


print("\nEvaluation:")

for key, value in results.items():

    print(
        key,
        ":",
        value
    )


# =========================================================
# SAVE MODEL
# =========================================================

print("\n======================================")

print("💾 SAVING MODEL")

print("======================================")


trainer.save_model(
    OUTPUT_DIR
)


processor.save_pretrained(
    OUTPUT_DIR
)


# =========================================================
# SAVE CLASS INFORMATION
# =========================================================

class_information = {

    "classes": [
        "REAL",
        "DEEPFAKE"
    ],

    "label2id": label2id,

    "id2label": id2label,

    "model": MODEL_NAME

}


with open(

    os.path.join(
        OUTPUT_DIR,
        "classes.json"
    ),

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        class_information,

        file,

        indent=4
    )


# =========================================================
# COMPLETE
# =========================================================

print("\n======================================")

print("✅ MODEL TRAINING COMPLETED")

print("======================================")


print("\nModel saved at:")

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


print("\nGenerated files:")

for file in os.listdir(
    OUTPUT_DIR
):

    print(
        "  ✅",
        file
    )


print("\nNext command:")

print(
    "streamlit run app.py"
)


print(
    "\n🤖 Deepfake AI Model is ready."
)
```
