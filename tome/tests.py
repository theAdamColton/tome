import unittest
import torch

from .tome import TokenMerger


class TestTome(unittest.TestCase):
    def test_constructor(self):
        r = 2
        s = 8
        k = torch.randn(4, s, 1)
        tm = TokenMerger(k, r)
        self.assertEqual(r, tm.dst_idx.size(1))

    def test_merge(self):
        r = 2
        s = 8
        k = torch.randn(4, s, 1)
        tm = TokenMerger(k, r)
        x = torch.randn(4, s, 16)
        tm(x)

    def test_merge_unmerge_with_ids(self):
        k = torch.randn(1, 32, 1)
        ids = torch.tensor(
            [[0] * 4 + [1] * 4 + [2] * 16 + [3] * 6 + [4] * 2], dtype=torch.long
        )
        tm = TokenMerger(k, 16, ids)
        merged_ids = tm.merged_ids
        unmerged_ids = tm.unmerge(merged_ids)
        self.assertTrue(torch.equal(ids, unmerged_ids))
