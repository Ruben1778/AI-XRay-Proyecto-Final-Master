import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from src.config import DROPOUT_RATE, IMG_SHAPE, LEARNING_RATE, NUM_CLASSES


def build_model(
    img_shape: tuple = IMG_SHAPE,
    num_classes: int = NUM_CLASSES,
    base_trainable: bool = False,
    fine_tune_at: int = None,
    dropout_rate: float = DROPOUT_RATE,
) -> tuple[Model, Model]:
    """Construye el modelo basado en ResNet50 con Transfer Learning.

    Retorna:
        tuple[Model, Model]: (model_completo, base_model)
    """
    # Cargar el modelo base ResNet50 con pesos preentrenados de ImageNet (sin la capa superior)
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=img_shape
    )

    # Fase A: Congelar el backbone si base_trainable es False
    base_model.trainable = base_trainable

    # Fase B: Si se especifica fine_tune_at, descongelar capas desde ese índice hacia adelante
    if base_trainable and fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True

    # Construir la cabeza clasificadora
    inputs = tf.keras.Input(shape=img_shape)
    x = base_model(inputs, training=base_trainable)
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(dropout_rate, name="dropout")(x)

    # Capa de salida (Sigmoid para clasificación binaria)
    if num_classes == 1:
        outputs = Dense(1, activation="sigmoid", name="predictions")(x)
    else:
        outputs = Dense(num_classes, activation="softmax", name="predictions")(
            x
        )

    model = Model(inputs, outputs, name="AI_XRay_ResNet50")
    return model, base_model


def compile_model(
    model: Model,
    learning_rate: float = LEARNING_RATE,
    num_classes: int = NUM_CLASSES,
) -> Model:
    """Compila el modelo con el optimizador, función de pérdida y métricas requeridas."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if num_classes == 1:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


if __name__ == "__main__":
    # Prueba de construcción del modelo congelado (Fase A)
    model, base_model = build_model(base_trainable=False)
    model = compile_model(model, learning_rate=1e-3)

    print(" Arquitectura construida exitosamente:")
    print(f"Total de capas en ResNet50: {len(base_model.layers)}")
    print(f"Parámetros totales: {model.count_params():,}")
    print(
        f"Parámetros entrenables (Fase A - Congelado): {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}"
    )

    # Prueba de Fine-Tuning (Fase B)
    model_ft, _ = build_model(
        base_trainable=True, fine_tune_at=143, dropout_rate=0.3
    )
    model_ft = compile_model(model_ft, learning_rate=1e-5)
    print(
        f"Parámetros entrenables (Fase B - Fine-Tuning): {sum([tf.keras.backend.count_params(w) for w in model_ft.trainable_weights]):,}"
    )
    
    
# ┌────────────────────────────────────────────────────────────────────────┐
# │ RUN                                                                    │
# └────────────────────────────────────────────────────────────────────────┘

# python -m src.architecture