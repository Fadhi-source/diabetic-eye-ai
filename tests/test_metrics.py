import numpy as np


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        from evaluation.metrics import compute_classification_metrics
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5, n_bootstrap=10)
        assert metrics["auc_roc"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0

    def test_threshold_changes_predictions(self):
        from evaluation.metrics import compute_classification_metrics
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.6, 0.6, 0.6, 0.6])
        strict = compute_classification_metrics(y_true, y_prob, threshold=0.7, n_bootstrap=10)
        loose = compute_classification_metrics(y_true, y_prob, threshold=0.5, n_bootstrap=10)
        assert strict["sensitivity"] <= loose["sensitivity"]

    def test_optimal_threshold_youden(self):
        from evaluation.metrics import optimal_threshold
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thresh, score = optimal_threshold(y_true, y_prob, method="youden")
        assert 0 < thresh < 1
        assert score > 0

    def test_optimal_threshold_f1(self):
        from evaluation.metrics import optimal_threshold
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thresh, score = optimal_threshold(y_true, y_prob, method="f1")
        assert 0 < thresh < 1
        assert score > 0

    def test_subgroup_analysis(self):
        from evaluation.metrics import subgroup_analysis
        import pandas as pd
        rng = np.random.default_rng(42)
        n = 40
        y_true = np.array([0]*30 + [1]*10)
        rng.shuffle(y_true)
        y_prob = np.clip(y_true + rng.normal(0, 0.3, n), 0, 1)
        groups = pd.DataFrame({"gender": [0, 1] * (n // 2)})
        result = subgroup_analysis(y_true, y_prob, groups)
        assert not result.empty
        assert "group_col" in result.columns


class TestPlotFunctions:
    def test_roc_pr_curves(self):
        from evaluation.metrics import plot_roc_pr_curves
        import tempfile, os
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        save_path = os.path.join(tempfile.gettempdir(), "test_roc_pr.png")
        fig = plot_roc_pr_curves(y_true, y_prob, save_path=save_path)
        assert fig is not None
        os.unlink(save_path)

    def test_calibration_curve(self):
        from evaluation.metrics import plot_calibration_curve
        import tempfile, os
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        save_path = os.path.join(tempfile.gettempdir(), "test_calibration.png")
        fig = plot_calibration_curve(y_true, y_prob, save_path=save_path)
        assert fig is not None
        os.unlink(save_path)
