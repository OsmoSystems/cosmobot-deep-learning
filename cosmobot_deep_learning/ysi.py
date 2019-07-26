import pandas as pd


def _guard_ysi_data_index_is_datetime(ysi_data):
    if not isinstance(ysi_data.index, pd.DatetimeIndex):
        raise ValueError(
            """
            ysi_data must be a DataFrame indexed by pre-parsed timestamps. Get this with, e.g.:
                pd.read_csv(
                    <ysi_data csv filename>,
                    parse_dates=['Timestamp']
                ).set_index('Timestamp')
            """
        )


def _guard_other_data_timestamp_column_is_datetime(
    other_data, other_data_timestamp_column
):
    if not pd.core.dtypes.common.is_datetime64_dtype(
        other_data[other_data_timestamp_column]
    ):
        raise ValueError(
            f"""
            other_data['{other_data_timestamp_column}'] column must have pre-parsed timestamps
            Get this with, e.g.:
                pd.read_csv(
                    <other data csv filename>,
                    parse_dates=['{other_data_timestamp_column}']
                )
            """
        )


def _guard_no_fractional_seconds(datetime_series, series_name):
    if any([t.microsecond for t in datetime_series]):
        raise ValueError(
            f"""
            {series_name} has fractional seconds.
            Data joining does not (currently) work if timestamps have fractional seconds.
            (If you get this error, ask a dev - we can fix it)
            """
        )


def join_interpolated_ysi_data(
    other_data,
    ysi_data,
    other_data_timestamp_column="timestamp",
    interpolation_method="slinear",
):
    """
    Params:
        other_data: DataFrame to be augmented with YSI data. Must have a pre-parsed timestamp column (datetime dtype).
        ysi_data: DataFrame from YSI. Should be indexed by pre-parsed timestamps (datetime dtype). Get this with, e.g.:
            pd.read_csv(
                <YSI filename>,
                parse_dates=['Timestamp']
            ).set_index('Timestamp')
        other_data_timestamp_column: Default: 'timestamp'. Column name in other_data containing timestamps.
        interpolation_method: Default: 'slinear'. Method used when interpolating YSI data.
            Passed to DataFrame.interpolate. 'slinear' is a 1st-order spline, conceptually equivalent to a linear
            interpolation.
    Return:
        DataFrame with each row in other_data augmented with the closest-timestamp data from the YSI.
        Discards "other_data" that is collected outside of the timerange of the YSI data.
        YSI columns will be prefixed with 'YSI ' (e.g. 'YSI Dissolved Oxygen (%)')
    """
    _guard_ysi_data_index_is_datetime(ysi_data)
    _guard_no_fractional_seconds(
        datetime_series=ysi_data.index, series_name="ysi_data.index"
    )

    _guard_other_data_timestamp_column_is_datetime(
        other_data, other_data_timestamp_column
    )
    _guard_no_fractional_seconds(
        datetime_series=other_data[other_data_timestamp_column],
        series_name=f"other_data['{other_data_timestamp_column}']",
    )

    # Upsample YSI data to allow any timestamp to be joined on a strict match
    resampled_ysi_data = (
        ysi_data.add_prefix("YSI ")
        .resample("s")
        .interpolate(method=interpolation_method)
    )
    return other_data.join(
        resampled_ysi_data,
        how="inner",  # Drop superfluous YSI rows and "other" rows outside of YSI timerange
        on=other_data_timestamp_column,
    ).reset_index(
        drop=True
    )  # Drop and reset the old "other" index, as some "other" rows may have been discarded
