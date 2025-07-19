import os
import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

attr_choose = ["age", "ethnic", "race", "hxcc", "hirsu", "bmi"]
attr_dim = 6
warnings.filterwarnings("ignore")


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        df = pd.read_csv(os.path.join(data_root, "Test.txt"), delim_whitespace=True)
        df["age"] = df["age"].astype(np.float32)
        df["hirsu"] = df["hirsu"].astype(np.float32)
        df["V32"] = df["V32"].astype(np.int64)
        df["V42"] = df["V42"].astype(np.float32)
        df["V45"] = df["V45"].astype(np.float32)
        df["V46"] = df["V46"].astype(np.float32)
        self.raw_data = df
        gt = pd.read_csv(os.path.join(data_root, "Label.txt"), delim_whitespace=True)

        # 自动识别变量类型
        self.continuous_cols = [
            col
            for col in df.columns
            if df[col].dtype in ["float32", "float64"] and col in attr_choose
        ]
        self.categorical_cols = [
            col
            for col in df.columns
            if df[col].dtype in ["int64", "object"] and col in attr_choose
        ]

        # 连续变量标准化
        self.scaler = StandardScaler()
        self.cont_data = self.scaler.fit_transform(df[self.continuous_cols]).astype(
            "float32"
        )

        # 离散变量 Label 编码
        self.cat_encoders = {}
        self.cat_data = []
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.cat_encoders[col] = le
            encoded = le.fit_transform(df[col])
            self.cat_data.append(encoded)
        self.cat_data = torch.tensor(list(zip(*self.cat_data)), dtype=torch.long)

        # 标签
        self.gt = gt.iloc[:, 0].values

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.cont_data[idx], dtype=torch.float32),  # 连续变量
            self.cat_data[idx],  # 离散变量（多个）
            self.gt[idx] - 1,  # 标签
        )

    def get_info(self):
        return len(self.categorical_cols), [
            len(enc.classes_) for enc in self.cat_encoders.values()
        ]


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        df = pd.read_csv(os.path.join(data_root, "Train.txt"), delim_whitespace=True)
        df["age"] = df["age"].astype(np.float32)
        df["hirsu"] = df["hirsu"].astype(np.float32)
        df["V32"] = df["V32"].astype(np.int64)
        df["V42"] = df["V42"].astype(np.float32)
        df["V45"] = df["V45"].astype(np.float32)
        df["V46"] = df["V46"].astype(np.float32)
        self.raw_data = df
        label_col = "PregnancyStatus"

        # 自动识别变量类型
        self.continuous_cols = [
            col
            for col in df.columns
            if df[col].dtype in ["float32", "float64"]
            and col != label_col
            and col in attr_choose
        ]
        self.categorical_cols = [
            col
            for col in df.columns
            if df[col].dtype in ["int64", "object"]
            and col != label_col
            and col in attr_choose
        ]

        # 连续变量标准化
        self.scaler = StandardScaler()
        self.cont_data = self.scaler.fit_transform(df[self.continuous_cols]).astype(
            "float32"
        )

        # 离散变量 Label 编码
        self.cat_encoders = {}
        self.cat_data = []
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.cat_encoders[col] = le
            encoded = le.fit_transform(df[col])
            self.cat_data.append(encoded)
        self.cat_data = torch.tensor(list(zip(*self.cat_data)), dtype=torch.long)

        # 标签
        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.cont_data[idx], dtype=torch.float32),  # 连续变量
            self.cat_data[idx],  # 离散变量（多个）
            self.labels[idx] - 1,  # 标签
        )

    def get_info(self):
        return len(self.categorical_cols), [
            len(enc.classes_) for enc in self.cat_encoders.values()
        ]


if __name__ == "__main__":
    train_dataset = TrainDataset(data_root="data")
    test_dataset = TestDataset(data_root="data")
    print("==== Data Overview ====")
    print(train_dataset.raw_data.head())

    print("\n==== Dataset Information ====")
    print(f"Count of continous columns: {len(train_dataset.continuous_cols)}")
    print(f"Continuous columns: {train_dataset.continuous_cols}")
    print(f"Count of categorical columns: {len(train_dataset.categorical_cols)}")
    print(f"Categorical columns: {train_dataset.categorical_cols}")
    print(f"Cardinalities of categorical columns: {train_dataset.get_info()[1]}")
