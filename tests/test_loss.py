import torch


class TestBinaryFocalLoss:
    def test_loss_positive(self, loss_fn):
        logits = torch.tensor([2.0, 1.5, 3.0, 0.8])
        targets = torch.ones(4)
        loss = loss_fn(logits, targets)
        assert loss > 0
        assert loss.item() < 1.0

    def test_loss_negative(self, loss_fn):
        logits = torch.tensor([-2.0, -1.5, -3.0, -0.8])
        targets = torch.zeros(4)
        loss = loss_fn(logits, targets)
        assert loss > 0
        assert loss.item() < 1.0

    def test_mixed_batch(self, loss_fn):
        logits = torch.tensor([2.0, -1.5, 0.3, -0.8])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss > 0

    def test_perfect_prediction_low_loss(self, loss_fn):
        logits = torch.tensor([100.0])
        targets = torch.ones(1)
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_wrong_prediction_high_loss(self, loss_fn):
        logits = torch.tensor([-100.0])
        targets = torch.ones(1)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.1

    def test_focal_loss_vs_bce_imbalanced(self):
        from training.loss import BinaryFocalLoss
        import torch.nn.functional as F

        focal = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(100)
        targets = torch.zeros(100)
        targets[:10] = 1.0

        focal_loss = focal(logits, targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        assert focal_loss < bce_loss, (
            "Focal loss should be lower than BCE on easy-negative-dominated data"
        )


class TestMultiLabelFocalLoss:
    def test_multi_label_shape(self):
        from training.loss import MultiLabelFocalLoss
        loss_fn = MultiLabelFocalLoss()
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()
        loss = loss_fn(logits, targets)
        assert loss > 0
