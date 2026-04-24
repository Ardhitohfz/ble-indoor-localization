import unittest

import pandas as pd

from ml_pipeline.scripts.data_validation import validate_csv_schema
from ml_pipeline.config_ml import REQUIRED_COLUMNS


class TestDataValidationSchema(unittest.TestCase):
    def test_validate_csv_schema_passes_when_required_columns_exist(self) -> None:
        df = pd.DataFrame([{column: 1 for column in REQUIRED_COLUMNS}])
        result = validate_csv_schema(df)
        self.assertTrue(result.passed)

    def test_validate_csv_schema_fails_when_columns_missing(self) -> None:
        columns = [column for column in REQUIRED_COLUMNS if column != REQUIRED_COLUMNS[0]]
        df = pd.DataFrame([{column: 1 for column in columns}])
        result = validate_csv_schema(df)
        self.assertFalse(result.passed)
        self.assertIn("Missing required columns", result.message)


if __name__ == "__main__":
    unittest.main()
