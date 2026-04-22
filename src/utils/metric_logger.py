from collections import defaultdict, deque

import torch


class SmoothedValue:
    """Track a series of values and provide smoothed statistics."""

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)


class EMAMetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.meters[key].update(value)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {attr!r}")

    def __str__(self):
        values = []
        for name, meter in self.meters.items():
            values.append(f"{name}: {meter.median:.4f} ({meter.global_avg:.4f})")
        return self.delimiter.join(values)


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self._dict = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self._dict[key] = float(value)

    def __getattr__(self, attr):
        if attr in self._dict:
            return self._dict[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {attr!r}")

    def __str__(self):
        values = []
        for key, value in self._dict.items():
            values.append(f"{key}: {value:.4f}")
        return self.delimiter.join(values)
