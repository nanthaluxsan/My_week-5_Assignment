# Named Entity Recognition (NER) with BERT

This project uses a BERT model from the Hugging Face Transformers library for Named Entity Recognition (NER). The model is fine-tuned to identify entities such as names, organizations, and locations within input text. This guide covers saving, loading, and performing inference with the model.

## Table of Contents

- [Setup](#setup)
- [Training and Saving the Model](#training-and-saving-the-model)
- [Implement inference function](#inference)
- [Check with trainned model](#running-the-code)
- [Check with pre-trained model](#example-output)
- [ Conclusion]

## Setup

### Dataset Preparation for Training

This project prepares the CoNLL-2003 dataset for training a Named Entity Recognition (NER) model. The following code demonstrates how to load the dataset, tokenize it, and align the labels.

```python
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class NERDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }

def prepare_data():
    # Load the CoNLL-2003 dataset
    dataset = load_dataset("conll2003")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    # Get label list from dataset
    label_list = dataset["train"].features["ner_tags"].feature.names

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"  # Return PyTorch tensors
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Padding token
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # Ignore tokens that are part of the same word
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized_datasets, len(label_list), label_list
```

## Model Implementation

In this section, we implement the Named Entity Recognition (NER) model using the BERT architecture. We use the `BertForTokenClassification` class from the `transformers` library to create our model.

```python
from transformers import BertForTokenClassification
import torch.nn as nn

class NERModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            'bert-base-cased',
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
```

## Training and Saving the Model

### evalution Function

helps to generate validation loss

```python
import torch
from tqdm import tqdm

def evaluate_model_loss(model, eval_dataloader, device):
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(eval_dataloader)
    return avg_val_loss
```

### Training Loop

This section outlines how to implement the training loop for the Named Entity Recognition (NER) model. The training process includes training the model, evaluating its performance, and implementing early stopping.

````python
def train_model(model, train_dataloader, eval_dataloader, optimizer, scheduler, device, id2label, num_epochs=10, patience=3):
    model.to(device)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Training phase
        model.train()  # Set model to training mode
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch data to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Clear previously calculated gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Retrieve the loss from the model's output

            # Backward pass (compute gradients)
            loss.backward()

            # Update parameters and learning rate
            optimizer.step()
            scheduler.step()

            # Accumulate training loss
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Training loss: {avg_train_loss:.4f}")

        # Validation phase
        val_loss = evaluate_model_loss(model, eval_dataloader, device)
        print(f"Validation loss: {val_loss:.4f}")

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load the best model for final evaluation
    model.load_state_dict(torch.load("best_model.pt"))
    print("Best model loaded.")
## Evaluation Function

This section outlines how to implement the evaluation function for the Named Entity Recognition (NER) model. The evaluation function computes the average validation loss and generates a classification report for the model's predictions.


```python
from seqeval.metrics import classification_report

def evaluate_model(model, eval_dataloader, device, id2label):
    model.eval()  # Set model to evaluation mode
    true_labels = []
    predicted_labels = []
    total_val_loss = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass to get logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

            logits = outputs.logits  # Raw predictions from the model
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()

            # Align predictions and labels by filtering out padding tokens
            for i, label_seq in enumerate(labels):
                true_seq = []
                pred_seq = []
                for j, label_id in enumerate(label_seq):
                    if label_id == -100:  # Ignore padding tokens
                        continue
                    true_seq.append(id2label[label_id])
                    pred_seq.append(id2label[predictions[i][j]])
                true_labels.append(true_seq)
                predicted_labels.append(pred_seq)

    # Calculate average validation loss
    avg_val_loss = total_val_loss / len(eval_dataloader)
    print(classification_report(true_labels, predicted_labels))
````

### Main Function

The `main()` function serves as the entry point for the Named Entity Recognition (NER) model training and evaluation process. It loads the dataset, prepares the data loaders, initializes the model, and orchestrates the training and evaluation.

```python
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare dataset
    print("Loading and preparing dataset...")
    tokenized_datasets, num_labels, label_list = prepare_data()
    id2label = {i: label for i, label in enumerate(label_list)}
    print(id2label)

    # Convert to custom Dataset objects
    train_dataset = NERDataset(tokenized_datasets["train"])
    eval_dataset = NERDataset(tokenized_datasets["validation"])
    test_dataset = NERDataset(tokenized_datasets["test"])

    # Create data loaders
    batch_size = 32  # You can adjust this as needed
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    # Model initialization
    model = NERModel(num_labels=num_labels)
    model.to(device)

    # Define optimizer with better parameters
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
        eps=1e-8
    )

    # Learning rate scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training
    print("Starting training...")
    train_model(model, train_dataloader, eval_dataloader, optimizer, scheduler, device, id2label=id2label)  # Pass id2label here

    # Evaluation
    print("\nEvaluating on validation set...")
    evaluate_model(model, eval_dataloader, device, id2label)

    return model
```

### Running the Model

After defining the `main()` function, you can execute it to train and evaluate the Named Entity Recognition (NER) model. The model will be trained on the CoNLL-2003 dataset, and evaluation metrics will be printed for the validation set.

### Executing the Main Function

To run the model, you simply need to call the `main()` function. The following code snippet shows how to do this:

```python
if __name__ == "__main__":
    model = main()
```

## Saving the Model and Tokenizer

After training your Named Entity Recognition (NER) model, you can save both the model and the tokenizer to your Google Drive for later use. This is particularly useful for preserving your work and reloading the model without needing to retrain it.

### Saving to Google Drive

To save the model and tokenizer, you can use the following code snippet in your Colab notebook:

```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Save the model and tokenizer
model.save_pretrained("/content/drive/MyDrive/idp_bootcamp/week_5/bert_ner_model")  # Save the model
tokenizer.save_pretrained("/content/drive/MyDrive/idp_bootcamp/week_5/bert_ner_model")  # Save the tokenizer
```

## Inference with the Trained Model

Once you have trained your Named Entity Recognition (NER) model, you can perform inference to identify entities in new text. The following steps outline how to set up and execute inference.

### Inference Function

The inference process involves preparing the model, tokenizing the input text, and then predicting the named entities. Here’s how to implement the inference function in your code:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def prepare_inference(model_path=None):
    """Initialize tokenizer and load model for inference"""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    if model_path:
        # Load model from the specified path
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    else:
        model = AutoModelForTokenClassification.from_pretrained('bert-base-cased')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move the model to the device

    id2label = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC"
    }

    return tokenizer, id2label, model, device

def inference(text, model, tokenizer, id2label, device):
    """Perform NER inference on input text"""
    model.eval()  # Set model to evaluation mode

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Extract logits from model outputs

    # Convert predictions to labels
    predictions = torch.argmax(logits, dim=2)
    predicted_labels = predictions[0].cpu().numpy()

    # Align predictions with words
    tokens = tokenizer.tokenize(text)
    labeled_words = []

    # Convert ids to labels
    for token, label_id in zip(tokens, predicted_labels):
        label = id2label[label_id]
        labeled_words.append((token, label))

    return labeled_words

def print_entities(labeled_words):
    """Pretty print the labeled entities"""
    current_entity = None
    entity_text = []

    for word, label in labeled_words:
        if label == "O":
            if current_entity:
                print(f"{current_entity}: {' '.join(entity_text)}")
                current_entity = None
                entity_text = []
        elif label.startswith("B-"):
            if current_entity:
                print(f"{current_entity}: {' '.join(entity_text)}")
            current_entity = label[2:]  # Remove "B-" prefix
            entity_text = [word]
        elif label.startswith("I-"):
            if current_entity == label[2:]:
                entity_text.append(word)
            else:
                if current_entity:
                    print(f"{current_entity}: {' '.join(entity_text)}")
                current_entity = label[2:]
                entity_text = [word]

    if current_entity:
        print(f"{current_entity}: {' '.join(entity_text)}")
```

## Check with trainned model

```python
#First initialize with the path to your trained model
tokenizer, id2label, model, device = prepare_inference1("/content/drive/MyDrive/idp_bootcamp/week_5/bert_ner_model")  # Ensure the path points to the correct model directory

# Example texts to analyze
texts = [
    "John Smith works at Microsoft in Seattle and visited New York last summer.",
    "The European Union signed a trade deal with Japan in Brussels.",
    "Tesla CEO Elon Musk announced new features coming to their vehicles."
]

# Process each text
for text in texts:
    print("\nText:", text)
    print("Entities found:")
    results = inference(text, model, tokenizer, id2label, device)  # Pass device to inference
    print_entities(results)

```

## Check with pre-trained model

```python

# First initialize with the path to your trained model
tokenizer, id2label, model, device = prepare_inference1()  # Ensure the path points to the correct model directory

# Example texts to analyze
texts1 = [
    "John Smith works at Microsoft in Seattle and visited New York last summer.",
    "The European Union signed a trade deal with Japan in Brussels.",
    "Tesla CEO Elon Musk announced new features coming to their vehicles."
]

# Process each text
for text in texts1:
    print("\nText:", text)
    print("Entities found:")
    results = inference(text, model, tokenizer, id2label, device)  # Pass device to inference
    print_entities(results)
```

## Conclusion

This project successfully developed a Named Entity Recognition (NER) system using a BERT-based model, capable of identifying and classifying entities like persons, organizations, and locations from text. Through thorough dataset preparation and training, the model demonstrated strong performance in recognizing entities in various contexts. The evaluation metrics affirmed its effectiveness, while the inference function enables users to extract entities from any input text. Future work could involve enhancing the model for multilingual support, integrating more features, or applying transfer learning techniques. This NER system serves as a robust foundation for various natural language processing applications.
