import os

from lightning.pytorch.callbacks import ModelCheckpoint as ModelCheckpoint_


class ModelCheckpoint(ModelCheckpoint_):
    def on_validation_end(self, *args, **kwargs) -> None:
        super().on_validation_end(*args, **kwargs)
        os.makedirs(self.dirpath, exist_ok=True)
        self.to_yaml()
