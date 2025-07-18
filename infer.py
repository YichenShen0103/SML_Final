import torch
import pandas as pd

from network import Classifier
from dataset import TestDataset


def infer(model, test_loader, device):
    results = []
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, gt in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
                prob = torch.sigmoid(output)
                predicted = (prob >= 0.5).float()
                total += gt.size(0)
                correct += (predicted == gt).sum().item()
                results.extend(predicted.cpu().numpy().flatten())
        print(f"Accuracy: {correct / total * 100:.2f}%")

    return [int(x) for x in results]  # Convert predictions to integers


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize test dataset and loader
    test_dataset = TestDataset(data_root="data")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    # Initialize model
    input_size = test_dataset[0][0].shape[0]  # 自动获取输入特征维度
    model = Classifier(input_size).to(device)

    # Load the trained model
    model.load_state_dict(torch.load("checkpoint/best_model.pth"))
    model.eval()

    # Perform inference
    predictions = infer(model, test_loader, device)

    # Save predictions to submission file
    sample_submission = pd.read_csv("data/sample_submission.csv")
    sample_submission["class"] = predictions
    sample_submission.to_csv("data/submission.csv", index=False)


if __name__ == "__main__":
    main()
