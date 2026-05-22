import torch
import pytest


@pytest.fixture
def dummy_batch():
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    tabular = torch.randn(batch_size, 17)
    labels = torch.randint(0, 2, (batch_size,)).float()
    return images, tabular, labels


@pytest.fixture
def model():
    from models.multimodal_model import MultiModalModel
    return MultiModalModel(pretrained=False)


@pytest.fixture
def loss_fn():
    from training.loss import BinaryFocalLoss
    return BinaryFocalLoss()
