import os
from typing import Optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tensorflow import keras  # type: ignore


def load_regular_model(base_dir: Optional[str] = None):
    """
    Load the regular policy model from keras_models/regular_model.keras.
    Raises FileNotFoundError with a helpful message if missing.
    """
    base = base_dir or os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base, 'keras_models', 'regular_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Regular model not found at {model_path}. Train it via `python -m honeybee_py train-regular`."
        )
    # Lazy import to avoid hard dependency at package import time
    import tensorflow as tf  # type: ignore
    return tf.keras.models.load_model(model_path)


def load_hornet_model(base_dir: Optional[str] = None, fallback_to_regular: bool = True):
    """
    Load the hornet confrontation model from keras_models/hornet_model.keras.
    If missing and fallback_to_regular=True, falls back to the regular model.
    """
    base = base_dir or os.path.dirname(os.path.dirname(__file__))
    hornet_path = os.path.join(base, 'keras_models', 'hornet_model.keras')
    if os.path.exists(hornet_path):
        import tensorflow as tf  # type: ignore
        return tf.keras.models.load_model(hornet_path)
    if fallback_to_regular:
        return load_regular_model(base_dir=base)
    raise FileNotFoundError(
        f"Hornet model not found at {hornet_path}. Train it via `python -m honeybee_py train-hornet`."
    )


