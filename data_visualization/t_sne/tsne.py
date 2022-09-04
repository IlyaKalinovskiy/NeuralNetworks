from torch import nn


class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super().__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_dim)

    def forward(self, pij, i, j):
        # TODO: реализуйте вычисление матрицы сходства для точек отображения и расстояние Кульбака-Лейблера
        # pij - значения сходства между точками данных
        # i, j - индексы точек
        loss_kld = None
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
