import torch
import pytest
from tite.model.tite import TiteConfig, TiteForPreTraining
from tite.module import TiteModule
from tite.datasets import FineWebDataModule, TransformationCollator
from tite.transformation import TokenMask
from tite.utils.adamw import AdamWNoWeightDecayBiasNorm
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_train_equivalent(tmp_path):
    # Set seed
    seed_everything(42)

    # Model config (from tite-2-late.yaml)
    config = TiteConfig(
        vocab_size=30522,
        num_hidden_layers=12,
        kernel_sizes=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        strides=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        hidden_sizes=768,
        num_attention_heads=12,
        intermediate_sizes=3072,
        positional_embedding_type="rotary",
        rotary_interleaved=True,
        hidden_act="gelu_pytorch_tanh",
        norm_location="post",
        norm_type="layer",
        qk_norm=True,
    )
    model = TiteForPreTraining(
        config=config,
        enhanced_masked_auto_encoding=True,
        enhanced_causal_auto_encoding=False,
        bow_auto_encoding=True,
    )

    token_mask = TokenMask(mask_id=103, mask_prob=0.3)
    collator = TransformationCollator(
        text_keys=["text", "null"],
        max_length=512,
        token_transformations={"contrastive_learning": [token_mask]},
    )

    # DataModule (from datamodule-tite-contrastive.yaml)
    datamodule = FineWebDataModule(
        path="arrow",
        data_files={
            "train": "./HuggingFaceFW___fineweb-edu/default/0.0.0/*/fineweb-edu-train-*.arrow"
        },
        batch_size=128,
        num_workers=8,
        streaming=True,
        collator=collator,
    )

    optimizer = AdamWNoWeightDecayBiasNorm(model.parameters(), lr=1e-4)

    from tite.utils.lr_scheduler import SigmoidLRSchedulerWithLinearWarmup
    lr_scheduler = SigmoidLRSchedulerWithLinearWarmup(
        optimizer,
        num_warmup_steps=3000,
        final_value=0.02,
    )

    lightning_module = TiteModule(model=model, tokenizer=None)  # Add tokenizer if needed

    callbacks = [
        ModelCheckpoint(),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = Trainer(
        max_steps=200000,
        precision="bf16-mixed",
        callbacks=callbacks,
        enable_progress_bar=False,
        val_check_interval=50000,
        accumulate_grad_batches=2,
        default_root_dir=tmp_path,
        logger=False,  # Set up Wandb logger if needed
        devices=1,
        accelerator="gpu",
    )

    # Fit (run for just 1 step for test speed)
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=None)
