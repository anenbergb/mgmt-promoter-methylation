from lightning.pytorch.callbacks import ModelCheckpoint as ModelCheckpoint_


class ModelCheckpoint(ModelCheckpoint_):
    def on_validation_end(self, *args, **kwargs) -> None:
        super().on_validation_end(*args, **kwargs)
        self.to_yaml()
