from glob import glob
import einx
import matplotlib
import matplotlib.pyplot as plt
import math
from typing import Optional
from torch import nn
import torch
from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer, Dinov2SelfAttention
import tome
import transformers
from PIL import Image


model_url = "facebook/dinov2-small"
model = transformers.Dinov2Model.from_pretrained(model_url)
processor = transformers.AutoProcessor.from_pretrained(model_url)

r = 16

tm = None


def patched_attn_forward(
    self,
    hidden_states,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
):
    mixed_query_layer = self.query(hidden_states)
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    # ---- BEGIN CHANGES ----------
    # key layer mean along the head dim
    return (context_layer, key_layer.mean(1))
    # ---- END CHANGES ------------


Dinov2SelfAttention.forward = patched_attn_forward


# these are lists in order to allow them to be assigned to
# in the patched layer forward function
adm_obj = [None]
sequence_ids_obj = [None]


def patched_layer_forward(
    self,
    hidden_states: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
):

    self_attention_outputs = self.attention(
        self.norm1(
            hidden_states
        ),  # in Dinov2, layernorm is applied before self-attention
        head_mask,
        output_attentions=output_attentions,
    )

    attention_output = self_attention_outputs[0]

    attention_output = self.layer_scale1(attention_output)
    outputs = self_attention_outputs[
        1:
    ]  # add self attentions if we output attention weights

    # first residual connection
    hidden_states = self.drop_path(attention_output) + hidden_states

    # ---- BEGIN CHANGES ----------
    k = self_attention_outputs[1]
    # sequence ids: used to prevent the cls token from being merged
    cls_token_id = 1
    if sequence_ids_obj[0] is None:
        sequence_ids_obj[0] = torch.zeros(
            k.size(0), k.size(1), dtype=torch.long, device=k.device
        )
        # make all cls tokens a different sequence ID
        sequence_ids_obj[0][:, 0] = cls_token_id
    tm = tome.TokenMerger(k, r, sequence_ids=sequence_ids_obj[0], adm=adm_obj[0])

    adm_obj[0] = tm.adm
    sequence_ids_obj[0] = tm.merged_ids

    hidden_states = tm.merge(hidden_states)
    # ---- END CHANGES -----------

    # in Dinov2, layernorm is also applied after self-attention
    layer_output = self.norm2(hidden_states)
    layer_output = self.mlp(layer_output)
    layer_output = self.layer_scale2(layer_output)

    # second residual connection
    layer_output = self.drop_path(layer_output) + hidden_states

    outputs = (layer_output,) + outputs

    return outputs


Dinov2Layer.forward = patched_layer_forward

images = [Image.open(f) for f in glob("./images/*.jpg")]
inputs = processor(images, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

last_hidden_state = outputs["last_hidden_state"]


adm = adm_obj[0]
cluster_ids = adm.argmax(dim=1)
# make sure that the cls token was never merged with other tokens
cls_id = cluster_ids[0, 0]
assert (cluster_ids[0, 1:] == cls_id).sum() == 0

colormap = matplotlib.colormaps["hsv"]

# renormalize cluster_ids from [0.0 to 1.0]

token_colors = cluster_ids / cluster_ids.max()
token_colors = token_colors.cpu()
token_colors = colormap(token_colors)
# remove alpha channel
token_colors = token_colors[..., :3]

patch_size = model.config.patch_size
nph = processor.crop_size["height"] // patch_size
npw = processor.crop_size["width"] // patch_size

# remove cls token
token_colors = token_colors[:, 1:]

token_images = einx.rearrange(
    "b (nph npw) c -> b c nph npw", token_colors, nph=nph, npw=npw
)

token_images = torch.from_numpy(token_images)
token_images = torch.nn.functional.interpolate(
    token_images, size=(nph * patch_size, npw * patch_size), mode="nearest"
)

token_images = token_images.movedim(1, -1)  # move channel last

input_images = inputs["pixel_values"].cpu().float()
input_images = input_images.movedim(1, -1)

# un normalize
image_std = torch.tensor(processor.image_std)
input_images = einx.multiply("b h w c, c", input_images, image_std)
image_mean = torch.tensor(processor.image_mean)
input_images = einx.add("b h w c, c", input_images, image_mean)

for i, (token_image, image) in enumerate(zip(token_images, input_images)):
    fig, axg = plt.subplots(2, 1)

    axg[0].imshow(image)
    axg[1].imshow(token_image)

    plt.grid(False)
    axg[0].set_xticks([])
    axg[0].set_yticks([])
    axg[1].set_xticks([])
    axg[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"example_output_{i:03}.jpg", dpi=200)
