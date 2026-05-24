from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from tite.datasets import GLUEDataModule
from tite.glue_module import GlueModule
from tite.model import TiteModel
from tite.model.tokenizer import TiteTokenizer
import argparse

# model_name_or_path = "/mnt/ceph/storage/data-tmp/current/ya38dij/tite/wandb/run-20250726_095635-l0bbbp2r/files/huggingface_checkpoint"
parser = argparse.ArgumentParser()
parser.add_argument("--tasks", nargs="+", required=True, help="List of GLUE tasks to train on")
parser.add_argument("--model_name_or_path", type=str, default="webis/tite-2-late", help="Path to the pretrained model")
args = parser.parse_args()

tasks = args.tasks
model_name_or_path = args.model_name_or_path

# https://arxiv.org/pdf/2312.17482
EPOCHS_MAP = {
    "cola": 10,
    "sst2": 3,
    "mrpc": 10,
    "stsb": 10,
    "qqp": 5,
    "mnli": 3,
    "qnli": 10,
    "rte": 3,
}

for task in tasks:
    print(f"Training on {task}...")

    tite = TiteModel.from_pretrained(model_name_or_path)
    tokenizer = TiteTokenizer.from_pretrained(model_name_or_path)
    glue = GLUEDataModule(task=task, batch_size=32, tokenizer=tokenizer, streaming=False, num_workers=4)
    model = GlueModule(
        tite,
        tokenizer,
        glue.hparams.name,
    )
    epochs = EPOCHS_MAP[task]
    trainer = Trainer(precision="bf16-mixed", max_epochs=epochs, enable_checkpointing=False, check_val_every_n_epoch=epochs)
    trainer.fit(model, glue)
