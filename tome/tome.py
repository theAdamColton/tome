import einx
import torch


def expand_x_to_y(y, x):
    y_shape = y.shape
    x_shape = x.shape
    new_y_shape = list(y_shape) + list(x_shape[len(y_shape) :])
    for _ in range(len(new_y_shape) - len(y_shape)):
        y = y.unsqueeze(-1)
    y = y.expand(new_y_shape)
    return y


class TokenMerger:
    """
    merges and merges tokens, exploiting the token to token similarity in k.

    example:
    batch_size = 32
    sequence_length = 128
    hidden_size = 768
    r = 32
    keys = torch.randn(batch_size, sequence_length, hidden_size)
    tm = TokenMerger(k, r)
    x = torch.randn(batch_size, sequence_length, 64)
    merged_x = tm(x) # shape: batch_size, r, 64
    unmerged_x = tm.unmerge(merged_x) # shape: batch_size, sequence_length, 64
    """

    def __init__(self, k: torch.Tensor, r: int, sequence_ids=None, mask_id=-100):
        """
        k: token embeddings, can be the key values from attention.
            shape: batch, sequence length, z

        r: the number of tokens to remove.

        sequence_ids: Uniquely identifies each element of each sequence.
            Protects tokens that are not part of the same instance from being merged.
            Optional
            shape: batch, sequence length

        """
        sequence_length = k.shape[1]
        assert sequence_length // 2 >= r
        assert r > 0
        with torch.no_grad():
            k = k / k.norm(dim=-1, keepdim=True)
            # step 1.) Assign tokens to set A or set B
            a, b = k[:, ::2, :], k[:, 1::2, :]
            # step 2.) Draw one edge between each token in set A and the most similar token in set B
            scores = einx.dot("b s1 z, b s2 z -> b s1 s2", a, b)

            if sequence_ids is not None:
                a_ids, b_ids = sequence_ids[:, ::2], sequence_ids[:, 1::2]
                attention_mask = a_ids.unsqueeze(2) == b_ids.unsqueeze(1)
                # scores where ids are not equal should be -inf
                scores.masked_fill_(~attention_mask, torch.finfo(scores.dtype).min)
                pad_mask = (a_ids == mask_id).unsqueeze(2) & (
                    b_ids == mask_id
                ).unsqueeze(1)
                # makes pad tokens have high scores
                scores.masked_fill_(pad_mask, torch.finfo(scores.dtype).max)

            node_max, node_idx = einx.max("b s1 s2 -> b s1", scores)
            edge_idx = node_max.argsort(dim=-1, descending=True)  # shape: b s 1
            # step 3.) Keep the top r most similar edges
            unm_idx = edge_idx[:, r:]  # Unmerged Tokens
            src_idx = edge_idx[:, :r]  # Merged Tokens
            dst_idx = node_idx.gather(dim=1, index=src_idx)  # shape: b r 1

            if sequence_ids is not None:
                # Merges ids, but doesn't do any reduction on merged ids,
                # Simply takes the items from set b.
                # The reduction is uncessary because each set of merged ids is assumed to be identical,
                # meaning that ids are not merged with ids that are different.
                # An id 0 shouldn't be be merged with any id that is not 0.
                unm_ids = a_ids.gather(dim=1, index=unm_idx)
                a_ids = a_ids.gather(dim=1, index=src_idx)
                assert torch.equal(
                    b_ids,
                    b_ids.scatter(1, dst_idx, a_ids),
                ), "These ids should be equal. If this test fails it means that attention mask was not properly computed, \
                and or tokens were incorrectly merged over sequence-id boundaries."
                self.merged_ids = torch.cat((unm_ids, b_ids), 1)
            else:
                self.merged_ids = None

        self.unm_idx = unm_idx
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.unmerged_sequence_length = k.shape[1]
        self.r = r

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(x)

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: batch, sequence length, z

        Does bipartite merging as keyed by unm_idx and src_idx
        """

        a, b = x[:, ::2], x[:, 1::2]
        unm = a.gather(dim=1, index=expand_x_to_y(self.unm_idx, a))
        a = a.gather(dim=1, index=expand_x_to_y(self.src_idx, a))

        og_dtype = b.dtype
        b = b.float()
        a = a.float()

        # step 4.) Merge connected tokens

        b = b.scatter_reduce(
            dim=1,
            index=expand_x_to_y(self.dst_idx, b),
            src=a,
            reduce="mean",
        )

        b = b.to(og_dtype)

        # step 5.) Concatenate sets back together
        return torch.cat((unm, b), 1)

    def unmerge(self, x: torch.Tensor) -> torch.Tensor:
        unm_len = self.unm_idx.shape[1]
        unm, b = x[:, :unm_len], x[:, unm_len:]

        dst_idx = expand_x_to_y(self.dst_idx, b)

        a = b.gather(dim=1, index=dst_idx)

        out_shape = list(x.shape)
        out_shape[1] = self.unmerged_sequence_length

        out = torch.zeros(*out_shape, device=x.device, dtype=x.dtype)

        out[:, 1::2] = b
        unm_idx = 2 * expand_x_to_y(self.unm_idx, unm)
        out.scatter_(
            dim=1,
            index=unm_idx,
            src=unm,
        )

        src_idx = 2 * expand_x_to_y(self.src_idx, a)
        out.scatter_(
            dim=1,
            index=src_idx,
            src=a,
        )

        return out
