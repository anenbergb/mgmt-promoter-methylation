import datetime
import time
from collections import defaultdict, deque

import torch
from lightning.pytorch.callbacks.progress import TQDMProgressBar


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
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
        return self.total / self.count


class ProgressBar(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_time = SmoothedValue(window_size=100)
        self.end_time = time.time()

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        step = trainer.global_step + 1
        max_steps = trainer.max_steps
        items["iter"] = f"{step}/{max_steps}"

        if trainer.training:
            eta_seconds = self.iter_time.global_avg * (max_steps - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            items["ETA"] = eta_string
        return items

    def on_train_start(self, *args, **kwargs):
        super().on_train_start(*args, **kwargs)
        self.end_time = time.time()

    def on_train_batch_end(self, *args, **kwargs):
        batch_time = time.time() - self.end_time
        self.iter_time.update(batch_time)
        super().on_train_batch_end(*args, **kwargs)
        self.end_time = time.time()
