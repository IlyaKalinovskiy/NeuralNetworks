import random

import torch

from torch.autograd import Variable


def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    endpoints = []
    start = 0
    for stop in range(0, len(args[0]), n):
        if stop - start > 0:
            endpoints.append((start, stop))
            start = stop
    random.shuffle(endpoints)
    for start, stop in endpoints:
        yield [a[start:stop] for a in args]


class Wrapper:
    def __init__(self, model, cuda=True, epochs=1000, batchsize=1024):
        self.batchsize = batchsize
        self.epochs = epochs
        self.cuda = cuda
        self.model = model
        if cuda:
            self.model.cuda()
        # TODO: добавьте оптимизатор

    def fit(self, *args):
        self.model.train()
        if self.cuda:
            self.model.cuda()
        for epoch in range(self.epochs):
            total = 0.0
            for itr, datas in enumerate(chunks(self.batchsize, *args)):
                datas = [Variable(torch.from_numpy(data)) for data in datas]
                if self.cuda:
                    datas = [data.cuda() for data in datas]

                    # TODO: вычислите логиты модели и градиенты от datas
                    # datas = [pij, i, j]
                    # pij - значения сходства между точками данных
                    # i, j - индексы точек

            msg = "Train Epoch: {} \tLoss: {:.6e}"
            msg = msg.format(epoch, total / (len(args[0]) * 1.0))
            print(msg)
