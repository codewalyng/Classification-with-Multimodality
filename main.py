import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import matplotlib.pyplot as plt
from modele.modeles import MultimodalClassifier
from utilfiles.data_import import EmbeddingDataset
import pickle


def train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, num_mastercategory,
                       num_subcategory, num_article_type, num_epochs):
    history = {
        'train_loss_mastercategory': [], 'train_loss_subcategory': [], 'train_loss_article_type': [],
        'train_accuracy_mastercategory': [], 'train_accuracy_subcategory': [], 'train_accuracy_article_type': [],
        'val_loss_mastercategory': [], 'val_loss_subcategory': [], 'val_loss_article_type': [],
        'val_accuracy_mastercategory': [], 'val_accuracy_subcategory': [], 'val_accuracy_article_type': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss_mastercategory, total_loss_subcategory, total_loss_article_type = 0, 0, 0
        total_correct_mastercategory, total_correct_subcategory, total_correct_article_type = 0, 0, 0
        total_samples = 0

        for data in train_loader:
            text_embeddings = data['text_embedding'].to(device)
            image_embeddings = data['image_embedding'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()
            mastercategory_output, subcategory_output, article_type_output = model(text_embeddings, image_embeddings)
            labels_mastercategory, labels_subcategory, labels_article_type = labels.split(
                [num_mastercategory, num_subcategory, num_article_type], dim=1)

            loss_mastercategory = criterion(mastercategory_output, labels_mastercategory)
            loss_subcategory = criterion(subcategory_output, labels_subcategory)
            loss_article_type = criterion(article_type_output, labels_article_type)

            total_loss = loss_mastercategory + loss_subcategory + loss_article_type
            total_loss.backward()
            optimizer.step()

            total_loss_mastercategory += loss_mastercategory.item()
            total_loss_subcategory += loss_subcategory.item()
            total_loss_article_type += loss_article_type.item()

            sigmoid = torch.sigmoid
            total_correct_mastercategory += (sigmoid(mastercategory_output) > 0.5).eq(
                labels_mastercategory > 0.5).sum().item()
            total_correct_subcategory += (sigmoid(subcategory_output) > 0.5).eq(labels_subcategory > 0.5).sum().item()
            total_correct_article_type += (sigmoid(article_type_output) > 0.5).eq(
                labels_article_type > 0.5).sum().item()
            total_samples += labels.size(0)

        model.eval()
        val_loss_mastercategory, val_loss_subcategory, val_loss_article_type = 0, 0, 0
        val_correct_mastercategory, val_correct_subcategory, val_correct_article_type = 0, 0, 0
        val_samples = 0

        with torch.no_grad():
            for data in val_loader:
                text_embeddings = data['text_embedding'].to(device)
                image_embeddings = data['image_embedding'].to(device)
                labels = data['label'].to(device)

                mastercategory_output, subcategory_output, article_type_output = model(text_embeddings,
                                                                                       image_embeddings)
                labels_mastercategory, labels_subcategory, labels_article_type = labels.split(
                    [num_mastercategory, num_subcategory, num_article_type], dim=1)

                loss_mastercategory = criterion(mastercategory_output, labels_mastercategory)
                loss_subcategory = criterion(subcategory_output, labels_subcategory)
                loss_article_type = criterion(article_type_output, labels_article_type)

                val_loss_mastercategory += loss_mastercategory.item()
                val_loss_subcategory += loss_subcategory.item()
                val_loss_article_type += loss_article_type.item()
                print(f'val_loss_article_type = {val_loss_article_type}')

                val_correct_mastercategory += (sigmoid(mastercategory_output) > 0.5).eq(
                    labels_mastercategory > 0.5).sum().item()
                val_correct_subcategory += (sigmoid(subcategory_output) > 0.5).eq(labels_subcategory > 0.5).sum().item()
                val_correct_article_type += (sigmoid(article_type_output) > 0.5).eq(
                    labels_article_type > 0.5).sum().item()
                val_samples += labels.size(0)

        history['train_loss_mastercategory'].append(total_loss_mastercategory / len(train_loader))
        history['train_accuracy_mastercategory'].append(
            total_correct_mastercategory / (total_samples * num_mastercategory))
        history['train_loss_subcategory'].append(total_loss_subcategory / len(train_loader))
        history['train_accuracy_subcategory'].append(total_correct_subcategory / (total_samples * num_subcategory))
        history['train_loss_article_type'].append(total_loss_article_type / len(train_loader))
        history['train_accuracy_article_type'].append(total_correct_article_type / (total_samples * num_article_type))
        history['val_loss_mastercategory'].append(val_loss_mastercategory / len(val_loader))
        history['val_accuracy_mastercategory'].append(val_correct_mastercategory / (val_samples * num_mastercategory))
        history['val_loss_subcategory'].append(val_loss_subcategory / len(val_loader))
        history['val_accuracy_subcategory'].append(val_correct_subcategory / (val_samples * num_subcategory))
        history['val_loss_article_type'].append(val_loss_article_type / len(val_loader))
        history['val_accuracy_article_type'].append(val_correct_article_type / (val_samples * num_article_type))

    return history


def plot_training_history(history):
    epochs = range(1, len(history['train_loss_mastercategory']) + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss_mastercategory'], label='Master Category')
    plt.plot(epochs, history['train_loss_subcategory'], label='Sub Category')
    plt.plot(epochs, history['train_loss_article_type'], label='Article Type')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['val_loss_mastercategory'], label='Master Category')
    plt.plot(epochs, history['val_loss_subcategory'], label='Sub Category')
    plt.plot(epochs, history['val_loss_article_type'], label='Article Type')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_accuracy_mastercategory'], label='Master Category')
    plt.plot(epochs, history['train_accuracy_subcategory'], label='Sub Category')
    plt.plot(epochs, history['train_accuracy_article_type'], label='Article Type')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_accuracy_mastercategory'], label='Master Category')
    plt.plot(epochs, history['val_accuracy_subcategory'], label='Sub Category')
    plt.plot(epochs, history['val_accuracy_article_type'], label='Article Type')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("utilfiles/ready_data/data_loaders.pkl", 'rb') as file:
        data_loaders = pickle.load(file)

    train_loader, val_loader = data_loaders['train_loader'], data_loaders['val_loader']

    model = MultimodalClassifier(text_embedding_dim=1024, image_embedding_dim=2048, num_mastercategory=7,
                                 num_subcategory=45, num_article_type=141)
    model.to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    num_epochs = 100
    history = train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, 7, 45, 141, num_epochs)

    plot_training_history(history)

    torch.save(model.state_dict(), "model_state.pth")


if __name__ == "__main__":
    main()