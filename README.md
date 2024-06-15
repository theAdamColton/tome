# tome

This is an open reproduction of [Token Merging: Your ViT but Faster](https://github.com/facebookresearch/ToMe) by Meta research.

This code is meant to be simple and easy to apply to new models. Token merging is applied over consecutive layers to decrease the sequence length. ToMe was shown to be effective technique to speed up ViTs. Applying ToMe to a ViT requires no additional finetuning and can speedup throughput by about 2x. The accuracy is only slightly reduced over the baseline. 

# Visualizations

These were created using a Dinov2 small model using the `example_dinov2.py` script.

The top image is the original. The middle image shows the clusterings of merged tokens; each color is a unique cluster. These clusters were obtained using the key-to-key similarities of the DinoV2 ViT-S. The bottom image is the result of merging the pixel values using the cluster assignments.

![on beach](figures/example_output_003.jpg)
![dog](figures/example_output_002.jpg)
![vilnius](figures/example_output_004.jpg)
![timp](figures/example_output_006.jpg)
![talinn](figures/example_output_008.jpg)


# Usage
There is one main TokenMerger class.

You instantiate the TokenMerger class with a tensor of keys, which indicate the token-to-token similarity.
```python
batch_size = 32
sequence_length = 128
hidden_size = 768
r = 32

keys = torch.randn(batch_size, sequence_length, hidden_size)
tm = TokenMerger(keys, r)
```

You can then merge tokens using the TokenMerger object.

```python
x = torch.randn(batch_size, sequence_length, 64)
merged_x = tm(x) # shape: batch_size, sequence_length - r, 64
```

You can unmerge tokens back into their original sequence length.

```python
unmerged_x = tm.unmerge(merged_x) # shape: batch_size, sequence_length, 64
```

Multiple merges can be done and then reversed over multiple layers. You need to pass the TokenMerger.adm from the previous layer into the TokenMerger constructor of the next layer.

```python
x = torch.randn(1, 16, 2)
tm1 = TokenMerger(x, 4)
x_merged1 = tm1(x) # shape: (1, 12, 2)
tm2 = TokenMerger(x_merged1, 4, adm=tm1.adm) # pass adm to tm2
x_merged2 = tm2(x_merged1) # shape: (1, 8, 2)
rec_x = tm2.unmerge(x_merged2) # shape: (1, 16, 2)
```
