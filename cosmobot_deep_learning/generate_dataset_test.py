from datetime import datetime
import pkg_resources

import pandas as pd

from . import generate_dataset as module


class TestPrepYsiData:
    def test_generates_correct_data_frame(self):
        expected = pd.DataFrame(
            {
                "Timestamp": [
                    datetime(2019, 5, 27, 9, 7, 9),
                    datetime(2019, 5, 27, 9, 8, 9),
                    datetime(2019, 5, 29, 16, 47, 56),
                    datetime(2019, 5, 29, 16, 48, 56),
                ],
                "Barometer (mmHg)": [763.6, 763.6, 758.5, 758.5],
                "Dissolved Oxygen (%)": [98.7, 98.7, 79.5, 79.6],
                "Temperature (C)": [34.4, 34.4, 19.9, 19.8],
                "Dissolved Oxygen (mmHg)": [
                    157.969903,
                    157.969903,
                    126.390372,
                    126.549354,
                ],
            }
        ).set_index("Timestamp")

        actual = module._prep_ysi_data(
            raw_data_folder="test_fixtures",
            ysi_filenames=["YSI_data_snapshot_1.csv", "YSI_data_snapshot_2.csv"],
        )

        pd.testing.assert_frame_equal(actual, expected)


class TestEverything:
    # An integration test so that I can be sure the same dataset would still be generated
    def test_everything(self):
        actual = module.do_everything(
            raw_data_folder="datasets/raw_data",
            ysi_filenames=[
                "YSI_data_snapshot_2019_06_04.csv",
                "YSI_data_2_snapshot_2019_06_04.csv",
            ],
            calibration_log_filename="Passive_Calibration_Data_Collection_Log_manual_cleanup_2019_06_05.csv",
        )
        expected_dataset_csv = pkg_resources.resource_filename(
            "cosmobot_deep_learning", "test_fixtures/expected_osmo_ml_dataset.csv"
        )
        expected = pd.read_csv(expected_dataset_csv, parse_dates=["timestamp"])

        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
