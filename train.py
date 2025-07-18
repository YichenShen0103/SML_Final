import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import TrainDataset
from network import Classifier, MixedClassifier


def train(model, epochs, train_loader, val_loader, device, criterion, optimizer):
    best_val_loss = float("inf")
    patience = 100
    patience_counter = 0
    print(f"Training model for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for cont_x, cat_x, y in train_loader:
            cont_x, cat_x, y = cont_x.to(device), cat_x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(cont_x, cat_x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # scheduler.step()

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_total = 0
            val_correct = 0
            for cont_x, cat_x, y in val_loader:
                cont_x, cat_x, y = cont_x.to(device), cat_x.to(device), y.to(device)
                output = model(cont_x, cat_x)
                loss = criterion(output, y)
                val_loss += loss.item()

                val_total += y.size(0)
                _, predicted = torch.max(output, 1)
                correct = (predicted == y).sum().item()
                val_correct += correct

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch [{epoch+1}]/[{epochs}], Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Accuracy: {val_correct / val_total * 100:.2f}%"
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ".cache/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def main():
    # hyperparameters
    epochs = 500

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset
    print("Initializing dataset...")
    dataset = TrainDataset(data_root="data")

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize network
    print("Initializing network...")
    # input_size = dataset[0][0].shape[0]
    # model = Classifier(input_size).to(device)
    model = MixedClassifier(
        num_cont=len(dataset.continuous_cols),
        cat_cardinalities=dataset.get_info()[1],
    ).to(device)

    # criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 处理 Sigmoid
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5
    )  # Example optimizer
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=10, verbose=True
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=500, eta_min=1e-5
    # )

    # Print model architecture
    train(model, epochs, train_loader, val_loader, device, criterion, optimizer)


if __name__ == "__main__":
    main()
