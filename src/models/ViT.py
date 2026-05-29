from torch import nn
from transformers.models.vit.modeling_vit import ViTModel
from huggingface_hub import snapshot_download
import os


class ViT(nn.Module):
    """
    A pretrained ViT model followed by a fully connected layer for classification
    """

    def __init__(self, num_labels=20, **kwargs):
        super(ViT, self).__init__()

        model_id = "google/vit-base-patch16-224"

        # Define path to store the model within the same directory as this script (src/models)
        script_dir = os.path.dirname(__file__)
        local_dir = os.path.join(script_dir, "vit-base-patch16-224-local")

        # Only download the model if the local directory doesn't already exist.
        if not os.path.exists(local_dir):
            print(
                f"Local model not found. Downloading from '{model_id}' to '{local_dir}'..."
            )
            snapshot_download(repo_id=model_id, local_dir=local_dir)
            print("Download complete.")
        else:
            print(f"Found local model at '{local_dir}'. Loading from disk.")

        # Load the model from the local directory
        model_output = ViTModel.from_pretrained(local_dir)
        self.base = model_output[0] if isinstance(model_output, tuple) else model_output
        self.final = nn.Linear(self.base.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(
            outputs.last_hidden_state[:, 0]
        )  # only the output corresponding to the first token (usually the class token in transformers used in classification) is used

        return logits
