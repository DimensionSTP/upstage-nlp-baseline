from typing import Dict, Any

import torch
from torch import nn

from transformers import (
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
)


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
    ) -> None:
        super().__init__()
        if "bart" in pretrained_model_name:
            self.model = BartForConditionalGeneration.from_pretrained(
                pretrained_model_name,
                output_hidden_states=False,
            )
        elif "t5" in pretrained_model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name,
                output_hidden_states=False,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                output_hidden_states=False,
            )

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = self.model(**encoded)
        return output

    def generate(
        self,
        encoded: Dict[str, torch.Tensor],
        options: Dict[str, Any],
    ) -> torch.Tensor:
        output = self.model.generate(
            **{
                **encoded,
                **options,
            }
        )
        return output
