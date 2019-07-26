import datetime
import os
import pkg_resources

import pandas as pd

# import numpy as np
# from tqdm import tqdm_notebook

from .ysi import join_interpolated_ysi_data

ATMOSPHERIC_OXYGEN_FRACTION = 0.2096

DATA_LOCATION = "./datasets/raw_data/"

YSI_FILENAMES = [
    "YSI_data_snapshot_2019_06_04.csv",
    "YSI_data_2_snapshot_2019_06_04.csv",
]

YSI_FILEPATHS = [
    os.path.join(DATA_LOCATION, ysi_filename) for ysi_filename in YSI_FILENAMES
]


def _prep_ysi_data(raw_data_folder, ysi_filenames):
    """ Combines multiple YSI logs into a single file, drop irrelevant columns,
        and back-calculate partial pressure of DO

        Args:
            ysi_filepaths: A list of filepaths to ysi data exports
    """
    ysi_filepaths = [
        pkg_resources.resource_filename(
            "cosmobot_deep_learning",
            os.path.join(raw_data_folder, ysi_filename)
            # "cosmobot_deep_learning",
            # raw_data_folder + "/" + ysi_filename,
        )
        for ysi_filename in ysi_filenames
    ]
    print(ysi_filepaths)

    ysi_data = pd.concat(
        [
            pd.read_csv(ysi_filepath, parse_dates=["Timestamp"]).set_index("Timestamp")
            for ysi_filepath in ysi_filepaths
        ]
    )

    ysi_data.drop(columns=["Comment", "Site", "Folder", "Unit ID"], inplace=True)

    # Back-calculate partial pressure of dissolved oxygen
    ysi_data["Dissolved Oxygen (mmHg)"] = (
        ysi_data["Dissolved Oxygen (%)"]
        * 0.01
        * ATMOSPHERIC_OXYGEN_FRACTION
        * ysi_data["Barometer (mmHg)"]
    )

    return ysi_data


def do_everything(raw_data_folder, ysi_filenames, calibration_log_filename):
    ###
    # Combine YSI data files
    ###

    YSI_data = _prep_ysi_data(raw_data_folder, ysi_filenames)

    ###
    # Open calibration log data
    ###

    calibration_log_filepath = pkg_resources.resource_filename(
        "cosmobot_deep_learning",
        os.path.join(raw_data_folder, calibration_log_filename),
    )

    # This is a manually cleaned up version of the collection log
    # which uses the N2 start time of the next row as the end time of the current row
    CALIBRATION_LOG = pd.read_csv(
        calibration_log_filepath,
        header=0,
        parse_dates=["sweep_n2_reset_started", "sweep_started", "sweep_ended"],
    )

    CALIBRATION_LOG.drop(columns=["observations/notes"], inplace=True)
    CALIBRATION_LOG.head()

    ALL_EXPERIMENTS = CALIBRATION_LOG["experiment"].dropna().unique()
    ALL_EXPERIMENTS

    ###
    # Flatten multiple ROI entries to single row per image
    ###
    ROI_NAMES = ["OO DO patch Wet", "Type 1 Chemistry Hand Applied Dry", "Left Glass"]
    MSORM_TYPES = ["r_msorm", "g_msorm", "b_msorm"]

    def flatten_ROI_msorm_to_row(experiment_df):
        """
        Uses a pivot table to flatten multiple ROIs for the same image into one row.
        Returns a DataFrame with one row per image, and msorm values for a subset of ROIs.
        """
        filtered_experiment_data = experiment_df[experiment_df["ROI"].isin(ROI_NAMES)]

        # Pivot creates a heirachical multiindex data frame with an index for each 'values' column
        pivot = filtered_experiment_data.pivot(
            index="image", columns="ROI", values=MSORM_TYPES + ["timestamp"]
        )

        # There's one copy of the timestamp for each ROI due to the pivot, so pull out the top level index
        timestamps = pivot["timestamp"]
        pivot.drop("timestamp", axis=1, inplace=True)

        # Flatten the pivot table index
        pivot.columns = [" ".join(col[::-1]).strip() for col in pivot.columns.values]

        # Add a single timestamp column back in and make it the index
        # There's probably a cleaner way to do this...
        pivot["timestamp"] = timestamps[timestamps.columns[0]]

        return pivot

    ###
    # Join the experiment log to get sweep # within experiment
    ###

    # Via: https://stackoverflow.com/questions/43593554

    def get_sweep_number_from_experiment_log(experiment_df, experiment_log):
        df_A = experiment_log.assign(key=1)
        df_B = experiment_df.assign(key=1)

        # Create a row for every (sweep, image) combination and filter to where image timestamp falls between range
        df_merge = pd.merge(df_A, df_B, on="key").drop("key", axis=1)
        df_merge = df_merge.query(
            "timestamp >= sweep_started and timestamp <= sweep_ended"
        )

        return df_merge

    ###
    # Combine all image and sweep data into experiment-specific DataFrames
    ###

    def extract_valid_data_points(experiment):
        # All of the experiments have already been pre-processed into csvs.
        experiment_image_data_filepath = pkg_resources.resource_filename(
            "cosmobot_deep_learning",
            os.path.join(raw_data_folder, f"{experiment}__all_camera_data.csv"),
        )

        # Jaime comment: were you finding that this was a problem?
        # If not, might be better to let it explode ("fail fast") so that you can check if something else is going wrong
        # if not os.path.isfile(experiment_data_file):
        #     print(f"Could not find data file for {experiment}, skipping.")
        #     return

        experiment_data = pd.read_csv(
            experiment_image_data_filepath, parse_dates=["timestamp"]
        )

        # Get this experiment from the calibration log and reindex as sweep number
        experiment_log = CALIBRATION_LOG[
            CALIBRATION_LOG["experiment"] == experiment
        ].reset_index(drop=True)

        experiment_data = flatten_ROI_msorm_to_row(experiment_data)
        experiment_data.reset_index(level=0, inplace=True)
        experiment_data = get_sweep_number_from_experiment_log(
            experiment_data, experiment_log
        )

        # Drop "bad sweeps" and extraneous columns
        experiment_data.drop(
            experiment_data[experiment_data["drop_data"] == "Yes"].index, inplace=True
        )
        experiment_data.drop(
            columns=[
                "sweep_started",
                "sweep_ended",
                "sweep_n2_reset_started",
                "drop_data",
            ],
            inplace=True,
        )

        return experiment_data

    ###
    # Combine all experiment DataFrames and join with YSI data
    ###
    all_image_data = [
        extract_valid_data_points(experiment)
        for experiment in ALL_EXPERIMENTS
        # for experiment in tqdm_notebook(ALL_EXPERIMENTS)
    ]
    all_experiment_data = join_interpolated_ysi_data(
        pd.concat(all_image_data), YSI_data
    )

    all_experiment_data.head()

    SAVE_DATASET = False

    if SAVE_DATASET:
        dataset_filename = f'{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}_osmo_ml_dataset.csv'
        all_experiment_data.to_csv(dataset_filename, index=False)  # Drop row numbers
        print(dataset_filename)

    return all_experiment_data
