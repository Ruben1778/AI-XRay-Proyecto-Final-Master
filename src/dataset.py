import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from src.config import (
    DATA_DIR,
    MANIFEST_PATH,
    MANIFEST_SPLIT_PATH,
    SEED,
    SUBSET_SIZE_PER_CLASS,
    USE_SUBSET_DEMO,
)


def extract_patient_id(filename: str) -> str:
    """Extrae el ID del paciente a partir del nombre del archivo.

    Para PNEUMONIA suele tener el patrón: person100_bacteria_475.jpeg ->
    person100. Para NORMAL, no hay ID claro, por lo que se usa el nombre único
    del archivo.
    """
    match = re.search(r"(person\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return filename.split(".")[0]


def build_manifest(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    
    """Explora el directorio de imágenes y construye un DataFrame manifiesto."""
    image_paths = []

    # Busca imágenes tanto en data_dir directo como dentro de chest_xray o archive
    
    possible_roots = [
        data_dir,
        data_dir / "chest_xray",
        data_dir / "Pediatric Chest X-ray Pneumonia",
        data_dir.parent
        / "archive (2)"
        / "Pediatric Chest X-ray Pneumonia",  # Estructura superior
    ]

    found_root = None
    for root in possible_roots:
        if root.exists() and any(root.rglob("*.jpeg")):
            found_root = root
            break

    if not found_root:
        raise FileNotFoundError(
            f"No se encontraron imágenes .jpeg en las rutas inspeccionadas desde: {data_dir}"
        )

    print(f"Ruta de imágenes encontrada: {found_root}")

    for img_path in found_root.rglob("*.jpeg"):
        # Ignorar carpetas ocultas o de mac
        if img_path.name.startswith("."):
            continue

        label_str = img_path.parent.name.upper()
        if label_str in ["NORMAL", "PNEUMONIA"]:
            label = 0 if label_str == "NORMAL" else 1
            patient_id = extract_patient_id(img_path.name)

            image_paths.append(
                {
                    "filepath": str(img_path.resolve()),
                    "filename": img_path.name,
                    "label_name": label_str,
                    "label": label,
                    "patient_id": patient_id,
                }
            )

    df = pd.DataFrame(image_paths)
    print(f"Manifiesto construido con {len(df)} imágenes en total.")

    # Guardar manifiesto base
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    return df


def create_split(
    df: pd.DataFrame, use_demo_subset: bool = USE_SUBSET_DEMO
) -> pd.DataFrame:
    """Aplica el split 70/15/15 agrupado por patient_id para evitar leak."""
    if use_demo_subset:
        print(
            f"Aplicando subconjunto demo: {SUBSET_SIZE_PER_CLASS} por clase..."
        )
        df_normal = (
            df[df["label_name"] == "NORMAL"]
            .sample(
                n=min(SUBSET_SIZE_PER_CLASS, len(df[df["label_name"] == "NORMAL"])),
                random_state=SEED,
            )
        )
        df_pneumonia = (
            df[df["label_name"] == "PNEUMONIA"]
            .sample(
                n=min(
                    SUBSET_SIZE_PER_CLASS,
                    len(df[df["label_name"] == "PNEUMONIA"]),
                ),
                random_state=SEED,
            )
        )
        df = (
            pd.concat([df_normal, df_pneumonia])
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )

    # Split agrupado por paciente (70% train, 30% temp)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    train_idx, temp_idx = next(gss.split(df, groups=df["patient_id"]))

    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()

    # Split de temp en 15% val y 15% test (50/50 de temp)
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
    val_idx, test_idx = next(
        gss_val.split(df_temp, groups=df_temp["patient_id"])
    )

    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df_split = (
        pd.concat([df_train, df_val, df_test])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    df_split.to_csv(MANIFEST_SPLIT_PATH, index=False)
    print(" Split 70/15/15 generado y guardado en manifest_split.csv.")
    print(df_split["split"].value_counts())
    return df_split


def check_patient_leakage(df_split: pd.DataFrame) -> bool:
    """Verifica si existen pacientes compartidos entre train, val y test."""
    train_patients = set(df_split[df_split["split"] == "train"]["patient_id"])
    val_patients = set(df_split[df_split["split"] == "val"]["patient_id"])
    test_patients = set(df_split[df_split["split"] == "test"]["patient_id"])

    leak_train_val = train_patients.intersection(val_patients)
    leak_train_test = train_patients.intersection(test_patients)
    leak_val_test = val_patients.intersection(test_patients)

    has_leak = len(leak_train_val) > 0 or len(leak_train_test) > 0 or len(leak_val_test) > 0

    if has_leak:
        print(f"⚠️ ATENCIÓN: Se detectó fuga de pacientes!")
        print(f"Fuga Train-Val: {len(leak_train_val)}")
        print(f"Fuga Train-Test: {len(leak_train_test)}")
        print(f"Fuga Val-Test: {len(leak_val_test)}")
    else:
        print(
            " SUCCESS: Verificación exitosa. No hay fuga de datos entre splits a nivel paciente."
        )

    return has_leak


if __name__ == "__main__":
    df = build_manifest()
    df_split = create_split(df)
    check_patient_leakage(df_split)
    
    

# ┌────────────────────────────────────────────────────────────────────────┐
# │ RUN                                 │
# └────────────────────────────────────────────────────────────────────────┘

# python -m src.dataset


