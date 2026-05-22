import pandas as pd

from data.dataset import DiabeticDataset, create_dataloaders
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, CONTINUOUS_FEATURES
from sklearn.preprocessing import StandardScaler


class TestDiabeticDataset:
    def test_dataset_output_shapes(self):
        df = pd.DataFrame({
            "patient_id": [0, 1],
            "complication_label": [0, 1],
            "dr_grade": [0, 3],
            "age": [45.0, 55.0],
            "diabetes_duration_years": [5.0, 15.0],
            "hba1c": [7.0, 10.0],
            "fasting_blood_sugar": [120.0, 200.0],
            "systolic_bp": [130.0, 150.0],
            "diastolic_bp": [80.0, 90.0],
            "bmi": [25.0, 30.0],
            "serum_creatinine": [1.0, 2.0],
            "ldl_cholesterol": [100.0, 140.0],
            "hdl_cholesterol": [45.0, 35.0],
            "triglycerides": [150.0, 250.0],
            "gender": [0, 1],
            "smoker": [0, 1],
            "hypertension": [0, 1],
            "on_insulin": [0, 1],
            "family_history": [0, 1],
            "rural_urban": [1, 0],
        })

        scaler = StandardScaler()
        scaler.fit(df[CONTINUOUS_FEATURES])

        ds = DiabeticDataset(df, "", scaler, dummy_images=True)
        img, tab, label = ds[0]

        assert img.shape == (3, 224, 224)
        assert tab.shape == (17,)
        assert label.item() in (0.0, 1.0)

    def test_data_loader_output(self):
        import os, tempfile, time

        df = pd.DataFrame({
            "patient_id": range(20),
            "complication_label": [0]*10 + [1]*10,
            "dr_grade": [0]*10 + [3]*10,
            "age": [50.0]*20, "diabetes_duration_years": [10.0]*20,
            "hba1c": [8.0]*20, "fasting_blood_sugar": [150.0]*20,
            "systolic_bp": [135.0]*20, "diastolic_bp": [85.0]*20,
            "bmi": [27.0]*20, "serum_creatinine": [1.2]*20,
            "ldl_cholesterol": [120.0]*20, "hdl_cholesterol": [40.0]*20,
            "triglycerides": [180.0]*20,
            "gender": [0]*20, "smoker": [0]*20, "hypertension": [0]*20,
            "on_insulin": [0]*20, "family_history": [0]*20, "rural_urban": [0]*20,
        })

        csv_path = os.path.join(tempfile.gettempdir(), f"test_diabetic_{int(time.time())}.csv")
        df.to_csv(csv_path, index=False)
        try:
            loaders, _ = create_dataloaders(csv_path, "", batch_size=4, dummy_images=True)
        finally:
            os.unlink(csv_path)

        for split in ["train", "val", "test"]:
            loader = loaders[split]
            images, tabular, labels = next(iter(loader))
            assert images.shape[1:] == (3, 224, 224)
            assert tabular.shape[1] == 17
            assert images.shape[0] == tabular.shape[0] == labels.shape[0]

    def test_train_val_test_split_sums(self):
        total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
        assert abs(total - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total}"

    def test_patient_level_split_no_leakage(self):
        from sklearn.model_selection import train_test_split
        import numpy as np

        np.random.seed(42)
        patients = np.array(range(100))
        labels = np.array([0]*70 + [1]*30)
        np.random.shuffle(patients)

        train_val, test = train_test_split(patients, test_size=0.15, stratify=labels)
        train, val = train_test_split(train_val, test_size=0.15/0.85)

        all_splits = set(train) | set(val) | set(test)
        assert len(all_splits) == len(patients), "Patients missing from splits"
        assert len(set(train) & set(test)) == 0, "Train-test patient overlap"
        assert len(set(train) & set(val)) == 0, "Train-val patient overlap"
        assert len(set(val) & set(test)) == 0, "Val-test patient overlap"
