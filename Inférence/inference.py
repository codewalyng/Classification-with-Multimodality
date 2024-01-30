import torch
from tqdm import tqdm
from datasets import load_dataset
from utilfiles.data_import import bert_model, resnet_model, tokenizer, preprocess_image
import matplotlib.pyplot as plt
from modele.modeles import MultimodalClassifier


# Fonctions auxiliaires
def load_model(model_path, device):
    model = MultimodalClassifier(text_embedding_dim=1024, image_embedding_dim=2048,
                                 num_mastercategory=7, num_subcategory=45, num_article_type=141)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate_text_embeddings(texts, tokenizer, model, device):
    model.eval()
    text_embeddings = []
    for text in tqdm(texts, desc="Generating text embeddings"):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings.append(output.last_hidden_state[:, 0, :].cpu())
    return text_embeddings


def generate_image_embeddings(images, model, device):
    model.eval()
    image_embeddings = []
    for image in tqdm(images, desc="Generating image embeddings"):
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
        image_embeddings.append(output.cpu())
    return image_embeddings


def inference(model, text_embeddings, image_embeddings):
    mastercategory_outputs, subcategory_outputs, article_type_outputs = [], [], []
    for text_emb, image_emb in zip(text_embeddings, image_embeddings):
        # Ajouter une dimension si nécessaire
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(0)
        if image_emb.dim() == 2:
            image_emb = image_emb.unsqueeze(0)

        with torch.no_grad():
            mastercat, subcat, arttype = model(text_emb, image_emb)
        mastercategory_outputs.append(mastercat)
        subcategory_outputs.append(subcat)
        article_type_outputs.append(arttype)

    return mastercategory_outputs, subcategory_outputs, article_type_outputs




# Chargement du modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model("model_state.pth", device)

# Préparation des données
dataset = load_dataset("ashraq/fashion-product-images-small")
sample = dataset['train'][9:11]

print(sample)
# Combine text and images for generating embeddings
text = []
images = []

# Itérer sur la longueur de l'un des champs du dictionnaire sample
for i in range(len(sample['id'])):
    # Construire la chaîne de texte pour chaque élément
    combined_text = f"{sample['productDisplayName'][i]} {sample['gender'][i]} {sample['baseColour'][i]} {sample['season'][i]} {sample['usage'][i]}"
    text.append(combined_text)

    # Prétraiter chaque image
    processed_image = preprocess_image(sample['image'][i])
    images.append(processed_image)

# Générer des embeddings
text_embeddings = generate_text_embeddings(text, tokenizer, bert_model, device)
image_embeddings = generate_image_embeddings(images, resnet_model, device)

print(len(text_embeddings))
print(len(image_embeddings))
# Effectuer l'inférence
mastercategory_output, subcategory_output, article_type_output = inference(model, text_embeddings, image_embeddings)

# Traiter les prédictions
sigmoid = torch.sigmoid
mastercategory_predictions = [sigmoid(output) > 0.5 for output in mastercategory_output]
subcategory_predictions = [sigmoid(output) > 0.5 for output in subcategory_output]
article_type_predictions = [sigmoid(output) > 0.5 for output in article_type_output]

# Afficher les prédictions
for i in range(len(sample)):
    print(f"Échantillon {i}:")
    print("Master Category:", mastercategory_predictions[i].int())
    print("Sub Category:", subcategory_predictions[i].int())
    print("Article Type:", article_type_predictions[i].int())
    img = plt.imread(images[i])
    plt.imshow(img)
    plt.show()
