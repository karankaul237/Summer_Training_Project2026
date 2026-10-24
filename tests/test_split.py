import unittest

import pandas as pd

from partition.split import stratified_split


class StratifiedSplitTests(unittest.TestCase):
    def setUp(self):
        # Several rows intentionally repeat a feature vector, including one
        # vector associated with conflicting labels.
        self.df = pd.DataFrame({
            "feature_a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "feature_b": [10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "label_id":  [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "label_name": ["benign", "attack"] + ["benign"] * 4 + ["attack"] * 6,
            "client": ["test"] * 12,
        })

    def _feature_set(self, frame):
        return set(map(tuple, frame[["feature_a", "feature_b"]].to_numpy().tolist()))

    def test_identical_feature_vectors_never_cross_splits(self):
        train, val, test = stratified_split(self.df)
        feature_sets = [self._feature_set(part) for part in (train, val, test)]
        self.assertFalse(feature_sets[0] & feature_sets[1])
        self.assertFalse(feature_sets[0] & feature_sets[2])
        self.assertFalse(feature_sets[1] & feature_sets[2])
        # Both rows for feature vector (1, 10), despite different labels, stay together.
        locations = [
            name for name, part in zip(("train", "val", "test"), (train, val, test))
            if ((part["feature_a"] == 1) & (part["feature_b"] == 10)).any()
        ]
        self.assertEqual(len(locations), 1)

    def test_all_rows_are_preserved(self):
        train, val, test = stratified_split(self.df)
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))

    def test_split_is_deterministic(self):
        first = stratified_split(self.df)
        second = stratified_split(self.df)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)

    def test_empty_input(self):
        train, val, test = stratified_split(self.df.iloc[0:0])
        self.assertTrue(train.empty and val.empty and test.empty)


if __name__ == "__main__":
    unittest.main()
