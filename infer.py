import torch
import pandas as pd

from network import MixedClassifier
from dataset import TestDataset


def infer(model, test_loader, device):
    results = []
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for cont_x, cat_x, y in test_loader:
            cont_x, cat_x = cont_x.to(device), cat_x.to(device)
            with torch.no_grad():
                output = model(cont_x, cat_x)
                # prob = torch.sigmoid(output)
                # predicted = (prob >= 0.5).float()
                total += y.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == y).sum().item()
                results.extend(predicted.cpu().numpy().flatten())
        print(f"Accuracy: {correct / total * 100:.2f}%")

    return [int(x + 1) for x in results]  # Convert predictions to integers


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize test dataset and loader
    test_dataset = TestDataset(data_root="data")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    # Initialize model
    # input_size = test_dataset[0][0].shape[0]  # 自动获取输入特征维度
    # model = Classifier(input_size).to(device)
    model = MixedClassifier(
        num_cont=len(test_dataset.continuous_cols),
        cat_cardinalities=test_dataset.get_info()[1],
    ).to(device)

    # Load the trained model
    # model.load_state_dict(torch.load(".cache/best_model.pth"))
    model.load_state_dict(torch.load("checkpoint/final_LTS.pth", map_location=device))
    model.eval()

    # Perform inference
    predictions = infer(model, test_loader, device)

    # Save predictions to submission file
    sample_submission = pd.read_csv("data/sample_submission.csv")
    sample_submission["class"] = predictions
    sample_submission.to_csv("data/submission.csv", index=False)


if __name__ == "__main__":
    main()
