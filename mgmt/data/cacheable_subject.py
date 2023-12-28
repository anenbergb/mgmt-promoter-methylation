import torchio as tio
import pickle
import os
from loguru import logger

class CacheableSubject(tio.Subject):
    def __init__(self, *args, **kwargs):
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                message = 'Only one dictionary as positional argument is allowed'
                raise ValueError(message)

        cache_dir = kwargs.pop("cache_dir") 
        super().__init__(**kwargs)
        os.makedirs(cache_dir, exist_ok=True)
        self["cache_dir"] = cache_dir
        self.cache_path = os.path.join(cache_dir, f"{self.patient_id_str}.pkl")
        self.cached = False
        self.cache_loaded = False
    
    @classmethod
    def from_subject(cls, subject: tio.Subject, cache_dir: str = ""):
        return CacheableSubject(cache_dir=cache_dir, **dict(subject.items()))
    
    def cache(self):
        images = self.get_images_dict(intensity_only=False)
        with open(self.cache_path, "wb") as f:
            pickle.dump(images, f)
        for name in images.keys():
            self.remove_image(name)
        self.cached = True
    
    def load_cache(self):
        logger.info(f"Loading {self.cache_path}")
        with open(self.cache_path, "rb") as f:
            images = pickle.load(f)
        for name, image in images.items():
            self.add_image(image, name)
        self.cache_loaded = True
    
    def load(self):
        if self.cache_loaded:
            logger.info(f"Cache already loaded")
            return
        elif self.cached:
            self.load_cache()
        else:
            super().load()