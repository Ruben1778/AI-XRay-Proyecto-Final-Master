import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
import pandas as pd
from src.config import BATCH_SIZE, IMG_SIZE, SEED


def create_data_augmentation_layer() -> tf.keras.Sequential:
    """Crea la capa de aumento de datos aplicable únicamente en entrenamiento."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.1, seed=SEED),
            tf.keras.layers.RandomZoom(0.1, seed=SEED),
            tf.keras.layers.RandomContrast(0.1, seed=SEED),
        ],
        name="data_augmentation",
    )


def load_and_preprocess_image(
    filepath: str, label: int, is_training: bool = False
) -> tuple[tf.Tensor, int]:
    """Carga una imagen desde disco, la redimensiona y aplica preprocess_input de ResNet50."""
    image_bytes = tf.io.read_file(filepath)
    # Decodificar imagen JPEG/PNG a 3 canales RGB
    image = tf.io.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    image = tf.image.resize(image, IMG_SIZE)

    # Preprocesamiento específico para ResNet50 (conversión RGB a BGR y resta de medias de ImageNet)
    image = preprocess_input(image)
    return image, label


def create_dataset_from_manifest(
    df: pd.DataFrame, is_training: bool = False, batch_size: int = BATCH_SIZE
) -> tf.data.Dataset:
    """Construye un pipeline optimizado tf.data.Dataset a partir de un DataFrame del manifiesto."""
    filepaths = df["filepath"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # Mapear la función de carga y preprocesamiento de imágenes
    dataset = dataset.map(
        lambda x, y: tf.py_function(
            func=lambda fp, l: load_and_preprocess_image(
                fp.numpy().decode("utf-8"), l, is_training
            ),
            inp=[x, y],
            Tout=[tf.float32, tf.int64],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Asegurar la forma explícita de los tensores
    def set_shapes(img, lbl):
        img.set_shape((*IMG_SIZE, 3))
        lbl.set_shape(())
        return img, lbl

    dataset = dataset.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=SEED)
        augmentation = create_data_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    from src.config import MANIFEST_SPLIT_PATH

    if MANIFEST_SPLIT_PATH.exists():
        df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
        df_train = df_split[df_split["split"] == "train"]

        ds_train = create_dataset_from_manifest(df_train, is_training=True)
        for images, labels in ds_train.take(1):
            print(f"Batch de imágenes extraído correctamente.")
            print(f"Forma del tensor de imágenes: {images.shape}")
            print(f"Forma del tensor de etiquetas: {labels.shape}")
    else:
        print(
            "⚠️ No se encontró manifest_split.csv. Ejecuta primero 'python -m src.dataset'."
        )
        
# ┌────────────────────────────────────────────────────────────────────────┐
# │ RUN                                                                    │
# └────────────────────────────────────────────────────────────────────────┘

#python -m src.preprocessing

