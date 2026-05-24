import importlib
from pathlib import Path

import yaml

from tite.datasets.basehfdatamodule import BaseHFDataModule
from tite.datasets.collator import Collator, TransformationCollator
from tite.model.tokenizer import TiteTokenizer


def test_resume(tokenizer: TiteTokenizer) -> None:
    data_dir = Path(__file__).parent / "data"
    datamodule = BaseHFDataModule(
        path="csv",
        collator=TransformationCollator(
            tokenizer,
            text_keys=("text", None),
            max_length=8,
        ),
        batch_size=2,
        data_dir=None,
        seed=42,
        data_files={"train": str(data_dir / "dummy-text.csv")},
        num_workers=0,
        streaming=True,
    )
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()

    iterator = iter(dataloader)
    first_sample = next(iterator)
    state_dict = datamodule.state_dict()
    second_sample = next(iterator)
    assert (first_sample["input_ids"] != second_sample["input_ids"]).any()

    datamodule = BaseHFDataModule(
        path="csv",
        collator=TransformationCollator(
            tokenizer,
            text_keys=("text", None),
            max_length=8,
        ),
        batch_size=2,
        data_dir=None,
        seed=42,
        data_files={"train": str(data_dir / "dummy-text.csv")},
        num_workers=0,
        streaming=True,
    )
    datamodule.setup(stage="fit")
    datamodule.load_state_dict(state_dict)
    dataloader = datamodule.train_dataloader()

    new_second_sample = next(iter(dataloader))

    assert (second_sample["input_ids"] == new_second_sample["input_ids"]).all()


def load_transformations_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    collator_args = config["data"]["init_args"]["collator"]["init_args"]

    # Load token transformations (supporting head-specific lists)
    token_transformations = {}
    for head_list in collator_args.get("token_transformations", []):
        for head, transforms in head_list.items():
            token_transformations[head] = []
            for t in transforms:
                class_path = t["class_path"]
                module_name, class_name = class_path.rsplit(".", 1)
                cls = getattr(importlib.import_module(module_name), class_name)
                token_transformations[head].append(cls(**t.get("init_args", {})))

    # Load string transformations (for a specific head, e.g. "contrastive")
    string_transformations = {}
    for head_list in collator_args.get("string_transformations", []):
        for head, transforms in head_list.items():
            string_transformations[head] = []
            for t in transforms:
                class_path = t["class_path"]
                module_name, class_name = class_path.rsplit(".", 1)
                cls = getattr(importlib.import_module(module_name), class_name)
                string_transformations[head].append(cls(**t.get("init_args", {})))
    return string_transformations, token_transformations


def test_resume_with_yaml_transformations(
    tokenizer: TiteTokenizer, string_transformations, token_transformations
) -> None:
    data_dir = Path(__file__).parent / "data"
    datamodule = BaseHFDataModule(
        path="csv",
        collator=TransformationCollator(
            tokenizer,
            text_keys=("text", None),
            max_length=8,
            string_transformations=string_transformations,
            token_transformations=token_transformations,
        ),
        batch_size=2,
        data_dir=None,
        seed=42,
        data_files={"train": str(data_dir / "dummy-text.csv")},
        num_workers=0,
        streaming=True,
    )
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()

    iterator = iter(dataloader)
    first_sample = next(iterator)
    state_dict = datamodule.state_dict()
    second_sample = next(iterator)
    assert (first_sample[0]["input_ids"] != second_sample[0]["input_ids"]).any()
    # assert (first_sample["input_ids"] != second_sample["input_ids"]).any()

    datamodule = BaseHFDataModule(
        path="csv",
        collator=TransformationCollator(
            tokenizer,
            text_keys=("text", None),
            max_length=8,
        ),
        batch_size=2,
        data_dir=None,
        seed=42,
        data_files={"train": str(data_dir / "dummy-text.csv")},
        num_workers=0,
        streaming=True,
    )
    datamodule.setup(stage="fit")
    datamodule.load_state_dict(state_dict)
    dataloader = datamodule.train_dataloader()

    new_second_sample = next(iter(dataloader))
    assert (second_sample[0]["input_ids"] == new_second_sample[0]["input_ids"]).all()
