from lightning import Callback
from lightning_ir.bi_encoder.bi_encoder_tokenizer import ADD_MARKER_TOKEN_MAPPING
from lightning_ir.modeling_utils.mlm_head import (
    MODEL_TYPE_TO_KEY_MAPPING,
    MODEL_TYPE_TO_LM_HEAD,
    MODEL_TYPE_TO_OUTPUT_EMBEDDINGS,
    MODEL_TYPE_TO_TIED_WEIGHTS_KEYS,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer

from tite.model.tite import BOWLMHead, TiteConfig, TiteModel
from tite.model.tokenizer import TiteTokenizer

AutoConfig.register(TiteConfig.model_type, TiteConfig)
AutoModel.register(TiteConfig, TiteModel)
AutoTokenizer.register(TiteConfig, None, TiteTokenizer)


# MODEL_TYPE_TO_LM_HEAD["tite"] = BOWLMHead

# MODEL_TYPE_TO_KEY_MAPPING["tite"] = {"heads.enhanced_masked_auto_encoding": "tite.projection"}
# # MODEL_TYPE_TO_KEY_MAPPING["tite"] = "this-is-a-dummy"

# MODEL_TYPE_TO_OUTPUT_EMBEDDINGS["tite"] = "lm_head.decoder"

# MODEL_TYPE_TO_TIED_WEIGHTS_KEYS["tite"] = ["lm_head.decoder.bias", "lm_head.decoder.weight"]

ADD_MARKER_TOKEN_MAPPING["tite"] = {
    "single": "{TOKEN} $0",
    "pair": "{TOKEN_1} $A [SEP] {TOKEN_2} $B:1",
}


class DummyImportCallback(Callback):
    pass
