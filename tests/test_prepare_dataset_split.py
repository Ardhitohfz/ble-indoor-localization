import unittest

import pandas as pd

from ml_pipeline.training.prepare_dataset import split_data


class TestSplitData(unittest.TestCase):
    def make_dataframe(self) -> pd.DataFrame:
        rows = []
        for i in range(20):
            rows.append(
                {
                    "ground_truth_cell": "A1",
                    "rssi_A": -60,
                    "rssi_B": -61,
                    "rssi_C": -62,
                    "rssi_D": -63,
                    "timestamp": f"2026-01-01T00:00:{i:02d}",
                }
            )
            rows.append(
                {
                    "ground_truth_cell": "B1",
                    "rssi_A": -70,
                    "rssi_B": -71,
                    "rssi_C": -72,
                    "rssi_D": -73,
                    "timestamp": f"2026-01-01T00:01:{i:02d}",
                }
            )
        return pd.DataFrame(rows)

    def test_split_data_requires_target_column(self) -> None:
        df = self.make_dataframe().drop(columns=["ground_truth_cell"])
        with self.assertRaises(ValueError):
            split_data(df)

    def test_split_data_requires_valid_ratio(self) -> None:
        df = self.make_dataframe()
        with self.assertRaises(ValueError):
            split_data(df, train_size=0.9, test_size=0.2)

    def test_split_data_keeps_class_coverage(self) -> None:
        df = self.make_dataframe()
        train_df, test_df = split_data(df, train_size=0.8, test_size=0.2, random_state=42)
        self.assertEqual(set(train_df["ground_truth_cell"].unique()), {"A1", "B1"})
        self.assertEqual(set(test_df["ground_truth_cell"].unique()), {"A1", "B1"})
        self.assertEqual(len(train_df) + len(test_df), len(df))


if __name__ == "__main__":
    unittest.main()
