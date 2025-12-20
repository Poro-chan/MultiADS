import os
import json


def extract_specie_from_filename(fname):
    name = fname.lower()
    if "_gl_" in name:
        return "glioma", "glioma", 1
    elif "_me_" in name:
        return "meningioma", "meningioma", 1
    elif "_pi_" in name:
        return "pituitary", "pituitary", 1
    elif "_nt_" in name:
        return "no_tumor", "good", 0
    else:
        return "unknown", "unknown", 1


def generate_meta_json_brisc2025(dataset_root, output_path="meta.json"):

    cls_root = os.path.join(dataset_root, "classification_task")
    seg_root = os.path.join(dataset_root, "segmentation_task")

    meta_data = {"train": {}, "test": {}}

    # -------------------------------------------------------
    # TRAIN SPLIT - Only NO_TUMOR from classification_task/train
    # -------------------------------------------------------
    train_nt_dir = os.path.join(cls_root, "train", "no_tumor")
    meta_data["train"] = {"brain_MRI": []}

    if not os.path.exists(train_nt_dir):
        print(f"⚠ Missing folder: {train_nt_dir}")
    else:
        for fname in os.listdir(train_nt_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = f"classification_task/train/no_tumor/{fname}"

            meta_data["train"]["brain_MRI"].append({
                "img_path": img_path,
                "mask_path": "",
                "cls_name": "brain_MRI",
                "specie_name": "good",
                "anomaly": 0
            })

    # -------------------------------------------------------
    # TEST SPLIT - Tumor classes from segmentation_task/test
    # -------------------------------------------------------
    test_img_dir = os.path.join(seg_root, "test", "images")

    meta_data["test"] = {
        "brain_MRI": []
    }

    if not os.path.exists(test_img_dir):
        print(f"⚠ Missing folder: {test_img_dir}")
    else:
        for fname in os.listdir(test_img_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            cls_name, specie_name, anomaly = extract_specie_from_filename(fname)

            img_path = f"segmentation_task/test/images/{fname}"
            mask_name = fname.rsplit(".", 1)[0] + ".png"
            mask_path = f"segmentation_task/test/masks/{mask_name}"

            meta_data["test"]["brain_MRI"].append({
                "img_path": img_path,
                "mask_path": mask_path,
                "cls_name": "brain_MRI",
                "specie_name": specie_name,
                "anomaly": anomaly
            })

    # -------------------------------------------------------
    # TEST SPLIT — Add NO_TUMOR from classification_task/test/no_tumor
    # -------------------------------------------------------
    test_nt_dir = os.path.join(cls_root, "test", "no_tumor")

    if not os.path.exists(test_nt_dir):
        print(f"⚠ Missing folder: {test_nt_dir}")
    else:
        for fname in os.listdir(test_nt_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = f"classification_task/test/no_tumor/{fname}"

            meta_data["test"]["brain_MRI"].append({
                "img_path": img_path,
                "mask_path": "",
                "cls_name": "brain_MRI",
                "specie_name": "good",
                "anomaly": 0
            })

    # -------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------
    with open(output_path, "w") as f:
        json.dump(meta_data, f, indent=4)

    print(f"✅ meta.json created at {output_path}")


# RUN
generate_meta_json_brisc2025(
    "./data/brisc2025",
    "./data/brisc2025/meta.json"
)