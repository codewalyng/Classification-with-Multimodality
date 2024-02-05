import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import BertModel, BertTokenizer
from torchvision import models
import torch.nn as nn
from utilfiles.data_import import preprocess_image
import matplotlib.pyplot as plt
from modele.modeles import MultimodalClassifier

# Define constants for better readability
TEXT_EMBEDDING_DIM = 1024
IMAGE_EMBEDDING_DIM = 2048
NUM_MASTER_CATEGORIES = 7
NUM_SUB_CATEGORIES = 45
NUM_ARTICLE_TYPES = 141


class Identity(nn.Module):
    def forward(self, x):
        return x


weights = models.ResNet152_Weights.IMAGENET1K_V1
resnet_model = models.resnet152(weights=weights)
resnet_model.fc = Identity()
resnet_model = resnet_model.eval()

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Bert
bert_model = BertModel.from_pretrained('bert-large-uncased')
bert_model = bert_model.eval()

# Prepare the data
# Prepare the data
dataset = load_dataset("ashraq/fashion-product-images-small")
indices = [11]
new_train_dataset = dataset['train'].select(indices)
# Construire un nouveau DatasetDict avec ce dataset
dataset = DatasetDict({
    'train': new_train_dataset
})


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

# Ensure text_embeddings and image_embeddings are NumPy arrays
text_embeddings = np.array(text_embeddings)
image_embeddings = np.array(image_embeddings)
# Apply the desired operations
text_embeddings = torch.from_numpy(text_embeddings)
image_embeddings = torch.from_numpy(image_embeddings)

# Define a function for loading the model
model = MultimodalClassifier(
    text_embedding_dim=TEXT_EMBEDDING_DIM,
    image_embedding_dim=IMAGE_EMBEDDING_DIM,
    num_mastercategory=NUM_MASTER_CATEGORIES,
    num_subcategory=NUM_SUB_CATEGORIES,
    num_article_type=NUM_ARTICLE_TYPES
)


# Define a function for loading the model
def load_model(model_path, device):
    model = MultimodalClassifier(
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        num_mastercategory=NUM_MASTER_CATEGORIES,
        num_subcategory=NUM_SUB_CATEGORIES,
        num_article_type=NUM_ARTICLE_TYPES
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'model_state.pth'
model = load_model(model_path, device)

# Perform inference
mastercategory_output, subcategory_output, article_type_output = model(text_embeddings, image_embeddings)

# Threshold the predictions (exemple de seuillage à 0.5, ajustez selon les besoins)
sigmoid = torch.sigmoid
mastercategory_predictions = [sigmoid(output) > 0.5 for output in mastercategory_output]
subcategory_predictions = [sigmoid(output) > 0.5 for output in subcategory_output]
article_type_predictions = [sigmoid(output) > 0.5 for output in article_type_output]


# Parcourir chaque échantillon dans le dataset
for i, item in enumerate(dataset['train']):
    print(f"Échantillon {i}: {item['productDisplayName']}")

    # Afficher les prédictions
    print("Master Category:", mastercategory_predictions[i].int())
    print("Sub Category:", subcategory_predictions[i].int())
    print("Article Type:", article_type_predictions[i].int())