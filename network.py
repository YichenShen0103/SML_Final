import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size=[32, 16]):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1),
        )

    def forward(self, x):
        return self.net(x)


class MixedClassifier(nn.Module):
    def __init__(self, num_cont, cat_cardinalities, embed_dim=2, hidden_size=[32, 16]):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, min(embed_dim, (num_categories + 1) // 2))
                for num_categories in cat_cardinalities
            ]
        )
        embed_total_dim = sum(emb.embedding_dim for emb in self.embeddings)

        input_dim = num_cont + embed_total_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 2),  # 二分类
        )

    def forward(self, cont_x, cat_x):
        embed_x = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        embed_x = torch.cat(embed_x, dim=1)
        x = torch.cat([cont_x, embed_x], dim=1)
        return self.mlp(x)
