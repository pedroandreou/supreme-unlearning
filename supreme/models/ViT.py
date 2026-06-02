from torch import nn
from transformers.models.vit.modeling_vit import ViTModel
import os


class ViT(nn.Module):
    """
    A pretrained ViT model followed by a fully connected layer for classification
    """

    def __init__(self, num_labels=20, **kwargs):
        super(ViT, self).__init__()

        model_id = "google/vit-base-patch16-224"

        # Prefer a co-located local copy if one exists (handy for offline / cluster
        # runs); otherwise load straight from the Hugging Face Hub. `from_pretrained`
        # caches into ~/.cache/huggingface (writable even when SUPREME is installed
        # read-only) - so nothing is bundled in the wheel and the download only
        # happens the first time someone actually uses the ViT model.
        script_dir = os.path.dirname(__file__)
        local_dir = os.path.join(script_dir, "vit-base-patch16-224-local")
        source = local_dir if os.path.isdir(local_dir) else model_id
        print(f"Loading ViT base from '{source}'.")

        model_output = ViTModel.from_pretrained(source)
        self.base = model_output[0] if isinstance(model_output, tuple) else model_output
        self.final = nn.Linear(self.base.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(
            outputs.last_hidden_state[:, 0]
        )  # only the output corresponding to the first token (usually the class token in transformers used in classification) is used

        return logits
