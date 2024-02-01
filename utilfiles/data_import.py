# Import libraries
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from modele.modeles import resnet_model, bert_model
import pickle
import os


# Define text preprocessing function
def preprocess_text(text, tokenizers):
    """Preprocesses the text using BERT tokenizer."""
    inputs = tokenizers(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = inputs['input_ids'].squeeze(0)
    attention_mask = inputs['attention_mask'].squeeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# Define image preprocessing function
def preprocess_image(image):
    """Preprocesses the image for use in the model."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = image.convert("RGB")
    image = preprocess(image)
    return image


# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


# Create a new dataset combining embeddings with labels
class EmbeddingDataset(Dataset):
    def __init__(self, text_embeddings, image_embeddings, labels):
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        return {
            "text_embedding": torch.tensor(self.text_embeddings[idx], dtype=torch.float32),
            "image_embedding": torch.tensor(self.image_embeddings[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


dataset = load_dataset("ashraq/fashion-product-images-small")

# Label notarization
masterCategory_binarizer = LabelBinarizer()
subCategory_binarizer = LabelBinarizer()
articleType_binarizer = LabelBinarizer()

masterCategories = [d['masterCategory'] for d in dataset['train']]
subCategories = [d['subCategory'] for d in dataset['train']]
articleTypes = [d['articleType'] for d in dataset['train']]

num_mastercategory = len(set(masterCategories))
num_subcategory = len(set(subCategories))
num_articleType = len(set(articleTypes))

encoded_masterCategories = masterCategory_binarizer.fit_transform(masterCategories)
encoded_subCategories = subCategory_binarizer.fit_transform(subCategories)
encoded_articleTypes = articleType_binarizer.fit_transform(articleTypes)

encoded_labels = np.concatenate([encoded_masterCategories, encoded_subCategories, encoded_articleTypes], axis=1)

# Check if the pickle file already exists
pickle_file_path = "utilfiles/ready_data/data_loaders.pkl"
if os.path.exists(pickle_file_path):
    print("Pickle file already exists. Skipping data processing.")
else:
    if not os.path.exists("utilfiles/ready_data"):
        os.makedirs("utilfiles/ready_data")

    # Generate embeddings for text
    def generate_text_embeddings(texts, tokenizer, model):
        text_embeddings = []

        with torch.no_grad():
            for text in tqdm(texts, desc="Generating text embeddings"):
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
                input_ids = inputs['input_ids'].to(model.device)
                attention_mask = inputs['attention_mask'].to(model.device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())

        return text_embeddings

    # Generate embeddings for images
    def generate_image_embeddings(images, model):
        image_embeddings = []

        # Determine the device of the model
        device = next(model.parameters()).device

        with torch.no_grad():
            for image in tqdm(images, desc="Generating image embeddings"):
                # Send image to the same device as the model
                image = image.to(device)
                output = model(image.unsqueeze(0))
                image_embeddings.append(output.squeeze(0).cpu().numpy())

        return image_embeddings


    # Combine text and images for generating embeddings
    texts = []
    for item in dataset['train']:
        combined_text = f"{item['productDisplayName']} {item['gender']} {item['baseColour']} {item['season']} {item['usage']}"
        texts.append(combined_text)

        images = [preprocess_image(item['image']) for item in dataset['train']]

    # Generate text and image embeddings
    text_embeddings = generate_text_embeddings(texts, tokenizer, bert_model)
    image_embeddings = generate_image_embeddings(images, resnet_model)
    embedding_dataset = EmbeddingDataset(text_embeddings, image_embeddings, encoded_labels)

    # Split the new embedding dataset into training, validation, and test sets
    train_indices, test_indices = train_test_split(range(len(embedding_dataset)), test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

    # Create data loaders for the downstream tasks using the embedding dataset
    train_loader = DataLoader(Subset(embedding_dataset, train_indices), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(embedding_dataset, val_indices), batch_size=32, shuffle=False)
    test_loader = DataLoader(Subset(embedding_dataset, test_indices), batch_size=32, shuffle=False)

    # Define a dictionary to store the data loaders
    data_loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader
    }

    # Save the data loaders to the "ready_data" folder using pickle
    with open("ready_data/data_loaders.pkl", "wb") as file:
        pickle.dump(data_loaders, file)
