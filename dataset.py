import os
import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

attr_choose = ["age", "ethnic", "race", "hxcc", "hirsu", "bmi"]
attr_dim = 6
warnings.filterwarnings("ignore")


def build_preprocessor(df: pd.DataFrame):
    # 初步区分离散和连续变量
    categorical_cols = []
    numerical_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            categorical_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() < 20:  # 可调节的阈值
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        elif pd.api.types.is_float_dtype(df[col]):
            numerical_cols.append(col)

    # 连续变量: 均值填补 + 标准化
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # 离散变量: 最频繁值填补 + OneHot编码
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 拼接预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # 拟合预处理器以便提取编码后的类别数
    preprocessor.fit(df)

    return preprocessor


# class TrainDataset(torch.utils.data.Dataset):
#     def __init__(self, data_root):
#         self.data_root = data_root

#         # 读取数据
#         df = pd.read_csv(
#             os.path.join(self.data_root, "Train.txt"), delim_whitespace=True
#         )
#         df["V32"] = df["V32"].astype(np.int32)
#         df["V42"] = df["V42"].astype(np.float64)

#         # 假设第一列是标签
#         self.labels = df.iloc[:, 0].values
#         self.features_df = df[attr_choose]  # 剩下的是特征

#         # 构建预处理器并拟合 + 转换
#         self.preprocessor = build_preprocessor(self.features_df)
#         self.features = self.preprocessor.fit_transform(self.features_df)
#         pca = PCA(n_components=attr_dim)
#         self.features_df = pca.fit_transform(self.features_df)

#         # 将标签映射到 [0, 1]
#         self.labels = self.labels.astype(int) - 1

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         x = torch.tensor(self.features[index], dtype=torch.float32)
#         y = torch.tensor(self.labels[index], dtype=torch.float32).view(
#             -1
#         )  # 确保标签是二维的
#         return x, y


# class TestDataset(torch.utils.data.Dataset):
#     def __init__(self, data_root):
#         self.data_root = data_root

#         # 读取数据
#         df = pd.read_csv(
#             os.path.join(self.data_root, "Test.txt"), delim_whitespace=True
#         )
#         df["V32"] = df["V32"].astype(np.int32)
#         df["V42"] = df["V42"].astype(np.float64)

#         self.features_df = df[attr_choose]

#         # 构建预处理器并拟合 + 转换
#         self.preprocessor = build_preprocessor(self.features_df)
#         self.features = self.preprocessor.fit_transform(self.features_df)
#         pca = PCA(n_components=attr_dim)
#         self.features_df = pca.fit_transform(self.features_df)

#         # 将标签映射到 [0, 1]
#         self.gt = pd.read_csv(
#             os.path.join(self.data_root, "Label.txt"), delim_whitespace=True
#         )
#         self.gt = self.gt.iloc[:, 0].values
#         self.gt = self.gt.astype(int) - 1

#     def __len__(self):
#         return len(self.gt)

#     def __getitem__(self, index):
#         x = torch.tensor(self.features[index], dtype=torch.float32)
#         y = torch.tensor(self.gt[index], dtype=torch.float32).view(
#             -1
#         )  # 确保标签是二维的
#         return x, y


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        df = pd.read_csv(os.path.join(data_root, "Test.txt"), delim_whitespace=True)
        df["age"] = df["age"].astype(np.float32)
        df["hirsu"] = df["hirsu"].astype(np.float32)
        df["V32"] = df["V32"].astype(np.int32)
        df["V42"] = df["V42"].astype(np.float32)
        df["V45"] = df["V45"].astype(np.float32)
        df["V46"] = df["V46"].astype(np.float32)
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
        df["V32"] = df["V32"].astype(np.int32)
        df["V42"] = df["V42"].astype(np.float32)
        df["V45"] = df["V45"].astype(np.float32)
        df["V46"] = df["V46"].astype(np.float32)
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
