from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import BertModel

# Bert
bert_model = BertModel.from_pretrained('bert-large-uncased')
bert_model = bert_model.eval()
# Resnet
weights = models.ResNet152_Weights.IMAGENET1K_V1
resnet_model = models.resnet152(weights=weights)


class MultimodalClassifier(nn.Module):
    def __init__(self, text_embedding_dim, image_embedding_dim, num_mastercategory, num_subcategory, num_article_type):
        """
        Initializes the MultimodalClassifier.

        Parameters:
        text_embedding_dim (int): Dimension of the text embeddings.
        image_embedding_dim (int): Dimension of the image embeddings.
        num_mastercategory (int): Number of master categories.
        num_subcategory (int): Number of subcategories.
        num_article_type (int): Number of article types.
        """
        super(MultimodalClassifier, self).__init__()
        combined_dim = text_embedding_dim + image_embedding_dim

        # MLP layers
        self.fc1 = nn.Linear(combined_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        # Output layers for each label category
        self.mastercategory_layer = nn.Linear(128, num_mastercategory)
        self.subcategory_layer = nn.Linear(128, num_subcategory)
        self.article_type_layer = nn.Linear(128, num_article_type)

    def forward(self, text_embeddings, image_embeddings):
        """
        Forward pass of the model.

        Parameters:
        text_embeddings (Tensor): Embeddings from text input.
        image_embeddings (Tensor): Embeddings from image input.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: Output from the three categories (mastercategory, subcategory, article_type).
        """
        text_embeddings = text_embeddings.squeeze(1)
        x = torch.cat((text_embeddings, image_embeddings), dim=1)

        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)

        mastercategory_output = self.mastercategory_layer(x)
        subcategory_output = self.subcategory_layer(x)
        article_type_output = self.article_type_layer(x)

        return mastercategory_output, subcategory_output, article_type_output
