import torch


class TestMultiModalModel:
    def test_forward_shape(self, model, dummy_batch):
        images, tabular, _ = dummy_batch
        out = model(images, tabular)
        assert out["probs"].shape == (4, 1)
        assert out["logits"].shape == (4, 1)
        assert out["img_emb"].shape == (4, 256)
        assert out["tab_emb"].shape == (4, 128)
        assert out["gate_weights"].shape == (4, 192)

    def test_probs_in_range(self, model, dummy_batch):
        images, tabular, _ = dummy_batch
        out = model(images, tabular)
        assert out["probs"].min() >= 0.0
        assert out["probs"].max() <= 1.0

    def test_predict_proba(self, model, dummy_batch):
        images, tabular, _ = dummy_batch
        probs = model.predict_proba(images, tabular)
        assert probs.shape == (4, 1)

    def test_fine_tune_mode(self, model):
        old_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.fine_tune_mode()
        new_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert new_trainable >= old_trainable

    def test_parameter_count(self, model, dummy_batch):
        images, tabular, _ = dummy_batch
        out = model(images, tabular)
        params = model.count_parameters()
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["total"] >= params["trainable"]
        assert params["total"] >= params["frozen"]


class TestImageBranch:
    def test_output_shape(self):
        from models.image_branch import ImageBranch
        branch = ImageBranch(pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = branch(x)
        assert out.shape == (4, 256)

    def test_freezing(self):
        from models.image_branch import ImageBranch
        branch = ImageBranch(pretrained=False, freeze_ratio=0.7)
        all_params = list(branch.backbone.parameters())
        frozen = sum(1 for p in all_params if not p.requires_grad)
        total = len(all_params)
        assert frozen / total >= 0.6, f"Expected ~70% frozen, got {frozen}/{total}"


class TestTabularBranch:
    def test_output_shape(self):
        from models.tabular_branch import TabularBranch
        branch = TabularBranch()
        x = torch.randn(8, 17)
        out = branch(x)
        assert out.shape == (8, 128)


class TestGatedFusion:
    def test_fusion_shape(self):
        from models.fusion import GatedFusion
        fusion = GatedFusion()
        img_emb = torch.randn(4, 256)
        tab_emb = torch.randn(4, 128)
        out = fusion(img_emb, tab_emb)
        assert out.shape == (4, 192)

    def test_gate_weights_shape(self):
        from models.fusion import GatedFusion
        fusion = GatedFusion()
        img_emb = torch.randn(4, 256)
        tab_emb = torch.randn(4, 128)
        gates = fusion.get_gate_weights(img_emb, tab_emb)
        assert gates.shape == (4, 192)
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0
