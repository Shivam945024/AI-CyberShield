import os
import json
import random
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "google/vit-base-patch16-224"

CSV_FILE = "data/face_deepfake/face_deepfake_dataset.csv"

OUTPUT_DIR = "models/face_deepfake"

IMAGE_SIZE = 224

EPOCHS = 3

BATCH_SIZE = 4

LEARNING_RATE = 2e-5

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LABELS
# ============================================================

LABEL2ID = {
    "REAL": 0,
    "DEEPFAKE": 1
}

ID2LABEL = {
    0: "REAL",
    1: "DEEPFAKE"
}


# ============================================================
# DATASET
# ============================================================

class FaceDeepfakeDataset(Dataset):

    def __init__(self, dataframe, processor, training=False):

        self.df = dataframe.reset_index(drop=True)

        self.processor = processor

        self.training = training

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]

        label = row["label"].upper()

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # Basic augmentation
        # ----------------------------------------------------

        if self.training:

            # Horizontal flip
            if random.random() < 0.5:

                image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        encoded = self.processor(
            images=image,
            return_tensors="pt"
        )

        # Remove batch dimension
        encoded = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        encoded["labels"] = torch.tensor(
            LABEL2ID[label],
            dtype=torch.long
        )

        return encoded


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = (
        predictions == labels
    ).mean()

    return {
        "accuracy": float(accuracy)
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("AI-CyberShield Face Deepfake Training")
    print("=" * 60)

    # --------------------------------------------------------
    # Check CSV
    # --------------------------------------------------------

    if not os.path.exists(CSV_FILE):

        print("\nERROR: Dataset CSV not found.")

        print(
            f"Expected location:\n{CSV_FILE}"
        )

        return

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(CSV_FILE)

    required_columns = [
        "image_path",
        "label"
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                f"\nERROR: Missing column: {column}"
            )

            print(
                "CSV must contain:"
            )

            print(
                "id,image_path,label"
            )

            return

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    invalid_labels = set(
        df["label"]
    ) - set(LABEL2ID.keys())

    if invalid_labels:

        print(
            "\nERROR: Invalid labels found:"
        )

        print(invalid_labels)

        print(
            "\nAllowed labels:"
        )

        print(
            "real"
        )

        print(
            "deepfake"
        )

        return

    # --------------------------------------------------------
    # Check image files
    # --------------------------------------------------------

    valid_rows = []

    print("\nChecking image files...\n")

    for _, row in df.iterrows():

        image_path = str(
            row["image_path"]
        )

        if os.path.exists(image_path):

            valid_rows.append(row)

        else:

            print(
                f"WARNING: Image not found: "
                f"{image_path}"
            )

    if len(valid_rows) == 0:

        print(
            "\nERROR: No valid images found."
        )

        return

    df = pd.DataFrame(valid_rows)

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\nDataset Summary")
    print("-" * 40)

    print(
        f"Total images: {len(df)}"
    )

    print(
        f"REAL: "
        f"{sum(df['label'] == 'REAL')}"
    )

    print(
        f"DEEPFAKE: "
        f"{sum(df['label'] == 'DEEPFAKE')}"
    )

    # --------------------------------------------------------
    # Check both classes
    # --------------------------------------------------------

    if len(
        df[df["label"] == "REAL"]
    ) == 0:

        print(
            "\nERROR: No REAL images found."
        )

        return

    if len(
        df[df["label"] == "DEEPFAKE"]
    ) == 0:

        print(
            "\nERROR: No DEEPFAKE images found."
        )

        return

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    # For very small datasets
    # use at least one sample for validation.

    if len(df) >= 10:

        train_size = int(
            len(df) * 0.8
        )

        train_df = df.iloc[
            :train_size
        ].copy()

        val_df = df.iloc[
            train_size:
        ].copy()

    else:

        print(
            "\nWARNING:"
        )

        print(
            "Dataset is very small."
        )

        print(
            "Training will use most images."
        )

        train_df = df.iloc[
            :-2
        ].copy()

        val_df = df.iloc[
            -2:
        ].copy()

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    print("\nLoading ViT processor...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = FaceDeepfakeDataset(
        train_df,
        processor,
        training=True
    )

    val_dataset = FaceDeepfakeDataset(
        val_df,
        processor,
        training=False
    )

    print(
        f"\nTraining images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading ViT model...")

    model = AutoModelForImageClassification.from_pretrained(

        MODEL_NAME,

        num_labels=2,

        label2id=LABEL2ID,

        id2label=ID2LABEL,

        ignore_mismatched_sizes=True
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,

        weight_decay=0.01,

        logging_steps=1,

        save_strategy="epoch",

        eval_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="accuracy",

        greater_is_better=True,

        report_to="none",

        remove_unused_columns=False,

        fp16=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        processing_class=processor,

        compute_metrics=compute_metrics
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.train()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    evaluation = trainer.evaluate()

    print("\nEvaluation Results:")

    for key, value in evaluation.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    trainer.save_model(
        OUTPUT_DIR
    )

    processor.save_pretrained(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Save label information
    # --------------------------------------------------------

    classes = {

        "label2id": LABEL2ID,

        "id2label": {
            str(key): value
            for key, value in ID2LABEL.items()
        }

    }

    classes_file = os.path.join(
        OUTPUT_DIR,
        "classes.json"
    )

    with open(
        classes_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            classes,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nModel saved to:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print("\nExpected files:")

    print("config.json")
    print("model.safetensors")
    print("preprocessor_config.json")
    print("classes.json")

    print("\nYou can now run:")

    print(
        "streamlit run app.py"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
