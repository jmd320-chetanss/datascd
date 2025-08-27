from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
from pyspark.sql import functions as spf
from datetime import datetime
from typing import Literal
import logging

_empty_logger = logging.getLogger("my_empty_logger")
_empty_logger.addHandler(logging.NullHandler())

SPECIAL_COLS = ["_effective_from", "_effective_to", "_reason", "_active"]

SchemaChangeHandleStrategy = Literal["error", "update"]


def _are_keys_valid(
    df: DataFrame | ConnectDataFrame,
    key_cols: list[str],
) -> bool:

    # Check if key columns as composite key represent unique records
    composite_key_check_df = df.groupBy(key_cols).count().filter("count > 1")

    # If there are duplicate records, the count will not be 0
    return composite_key_check_df.count() == 0


def generate(
    spark_session: SparkSession,
    source_df: DataFrame | ConnectDataFrame,
    target_df: DataFrame | ConnectDataFrame | None,
    key_cols: list[str],
    schema_change_handle_strategy: SchemaChangeHandleStrategy = "update",
    log_analytics: bool = False,
    current_datetime: datetime = datetime.now(),
    logger: logging.Logger = _empty_logger,
):
    """
    Generates an SCD2 table from the source and target dataframes.
    If the target dataframe already exists, it updates the target dataframe to preserve
    the history and add any new modifications.

    Args:
        source_df: The source dataframe.
        target_df: The target dataframe. If None, a new dataframe will be created.
        key_cols: A list of columns that uniquely identify a record.
    """

    # ----------------------------------------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------------------------------------

    assert isinstance(
        source_df, (DataFrame, ConnectDataFrame)
    ), "source_df must be a DataFrame or ConnectDataFrame."

    assert isinstance(
        target_df, (DataFrame, ConnectDataFrame, type(None))
    ), "target_df must be a DataFrame, ConnectDataFrame, or None."

    assert isinstance(key_cols, list), "key_cols must be a list."

    assert len(key_cols) > 0, "key_cols must not be empty."

    assert all(
        isinstance(col, str) for col in key_cols
    ), "All elements in key_cols must be strings."

    assert all(col in source_df.columns for col in key_cols), \
        "All key_cols must be present in source_df."

    assert _are_keys_valid(
        df=source_df, key_cols=key_cols
    ), "Key columns represent duplicate records."

    assert not any(col in SPECIAL_COLS for col in source_df.columns), \
        f"Source dataframe must not contain '{SPECIAL_COLS}' columns."

    if target_df is not None:

        assert all(col in target_df.columns for col in key_cols), \
            "All key_cols must be present in target_df."

        assert all(col in target_df.columns for col in SPECIAL_COLS), \
            f"Target dataframe must contain '{SPECIAL_COLS}' columns."

    # ----------------------------------------------------------------------------------------------
    # SCD2 Generation
    # ----------------------------------------------------------------------------------------------

    logger.debug(f"Key cols: {key_cols}")
    logger.debug(f"Timestamp: {current_datetime}")
    logger.debug(f"Source cols: {source_df.columns}")

    if target_df is None:
        cols = source_df.columns

        target_df = spark_session.sql(
            f"""
            select
                {", ".join(cols)}
                , cast(null as timestamp) as _effective_from
                , cast(null as timestamp) as _effective_to
                , cast(null as string) as _reason
                , cast(null as boolean) as _active
            from {{source}}
            where false
            """,
            source=source_df,
        )

    else:
        logger.debug(f"Target cols: {target_df.columns}")

        cols = [col for col in source_df.columns if col in target_df.columns]

        # ------------------------------------------------------------------------------------------
        # Handle schema changes, new columns found in source not present in target
        # ------------------------------------------------------------------------------------------

        source_cols_not_in_target_cols = [
            col for col in source_df.columns
            if col not in target_df.columns
        ]

        if source_cols_not_in_target_cols:

            logger.info(f"New columns found: {source_cols_not_in_target_cols}")

            match schema_change_handle_strategy:
                case "error":
                    raise ValueError(
                        f"Source dataframe contains columns not present in target dataframe: {source_cols_not_in_target_cols}"
                        f"Hint: Use schema_change_handle_strategy='update' to add the new columns to the target dataframe."
                    )

                case "update":
                    logger.info("All new columns added to target dataframe.")

                    cols += source_cols_not_in_target_cols

                    target_df = target_df.withColumns(
                        {col: spf.lit(None) for col in source_cols_not_in_target_cols}
                    )

        # ------------------------------------------------------------------------------------------
        # Handle schema changes, columns found in target not present in source
        # ------------------------------------------------------------------------------------------

        target_cols_not_in_source_cols = [
            col for col in target_df.columns
            if col not in source_df.columns
            and not col in SPECIAL_COLS
        ]

        if target_cols_not_in_source_cols:

            logger.info(
                f"Deleted columns found: {target_cols_not_in_source_cols}")

            match schema_change_handle_strategy:
                case "error":
                    raise ValueError(
                        f"Target dataframe contains columns not present in source dataframe: {target_cols_not_in_source_cols}"
                        f"Hint: Use schema_change_handle_strategy='update' to hanlde the deleted columns from the source dataframe."
                    )

                case "update":
                    logger.info(
                        "All deleted columns added to source dataframe with null values.")

                    cols += target_cols_not_in_source_cols

                    source_df = source_df.withColumns(
                        {col: spf.lit(None) for col in target_cols_not_in_source_cols}
                    )

    # ----------------------------------------------------------------------------------------------
    # SCD2 Implementation
    # ----------------------------------------------------------------------------------------------

    logger.debug(f"Final cols: {cols}")

    non_key_cols = [col for col in cols if col not in key_cols]
    logger.debug(f"Non key cols: {non_key_cols}")

    # Sql query to generate target table with all the new records added,
    # updated records modified and added and deleted records marked deleted
    scd_sql = f"""
        with source as (
            select {", ".join(cols)} from {{in_source}}
        ),

        target as (
            select {", ".join(cols)},
                _effective_from,
                _effective_to,
                _reason,
                _active

            from {{in_target}}
        ),

        -- Records in source that are not present in target
        source_new as (
            select source.*
            from source
            left join target on {" and ".join([f"source.{col} = target.{col}" for col in key_cols])}
            where {" and ".join([f"target.{col} is null" for col in key_cols])}
        ),

        -- Records in source that are present in target and have been modified
        source_modified as (
            select source.*
            from source
            inner join target on {" and ".join([f"source.{col} = target.{col}" for col in key_cols])}
            -- compare against the active records only
            where target._active is true
                {" and (" + " or ".join([f"source.{col} is distinct from target.{col}" for col in non_key_cols]) + ")" if non_key_cols else ""}
        ),

        -- Records in target that are not present in source
        target_delete as (
            select target.*
            from target
            left join source on {" and ".join([f"source.{col} = target.{col}" for col in key_cols])}
            -- Compare against the active records only
            where target._active is true
                and ({" and ".join([f"source.{col} is null" for col in key_cols])})
        )

        -- Pick all the new records from source and mark them as new
        select
            *
            , cast('{current_datetime}' as timestamp) as _effective_from
            , cast(null as timestamp) as _effective_to
            , 'New' as _reason
            , true as _active
        from source_new

        union all

        -- Pick all the records from target that have been deleted and mark them as deleted
        select
            {", ".join(f"target.{col}" for col in cols)}
            , target._effective_from
            , cast('{current_datetime}' as timestamp) as _effective_to
            , 'Delete' as _reason
            , false as _active
        from target_delete
        inner join target on {" and ".join([f"target.{col} = target_delete.{col}" for col in key_cols])}

        union all

        -- Pick all the records from target that have been modified and mark them as deleted
        select
            {", ".join(f"target.{col}" for col in cols)}
            , target._effective_from

            -- if the old record has been deleted and a new record comes with the same id,
            -- dont't overwrite the end date and reason of the old record
            , coalesce(target._effective_to, cast('{current_datetime}' as timestamp)) as _effective_to
            , case when target._active is false then target._reason else 'Update' end as _reason

            , false as _active
        from source_modified
        inner join target on {" and ".join([f"target.{col} = source_modified.{col}" for col in key_cols])}

        union all

        -- Pick all the records from source that have been modified and mark them as updated
        select
            *
            , cast('{current_datetime}' as timestamp) as _effective_from
            , cast(null as timestamp) as _effective_to
            , 'Update' as _reason
            , true as _active
        from source_modified

        union all

        -- Pick all the remaining (old, unmodified and active) records from target
        select target.*
        from target
        left join source_modified on {" and ".join([f"target.{col} = source_modified.{col}" for col in key_cols])}
        left join target_delete on {" and ".join([f"target.{col} = target_delete.{col}" for col in key_cols])}
        where {" and ".join([f"source_modified.{col} is null" for col in key_cols])}
            and {" and ".join([f"target_delete.{col} is null" for col in key_cols])}
    """

    logger.debug(f"Sql query to generate SCD2 target: {scd_sql}")
    logger.info("Generating SCD2 target...")

    result_df = spark_session.sql(
        scd_sql,
        in_source=source_df,
        in_target=target_df,
    )

    logger.info("Generating SCD2 target done.")

    if log_analytics:
        logger.info("Collecting analytics...")

        new_record_sql = f"""
            select count(*) as count
            from {{result}}
            where _reason = 'New'
                and _effective_from = '{current_datetime}'
        """

        logger.debug(f"Sql query to find new records: {new_record_sql}")
        logger.info("Finding new records...")

        new_record_count = spark_session.sql(
            new_record_sql,
            result=result_df,
        ).collect()[0]["count"]

        logger.info(
            f"Finding new records done, found '{new_record_count}' new records."
        )

        modified_record_sql = f"""
            select count(*) as count
            from {{result}}
            where _reason = 'Update'
                and _effective_from = '{current_datetime}'
        """

        logger.debug(
            f"Sql query to find modified records: {modified_record_sql}")
        logger.info("Finding modified records...")

        modified_record_count = spark_session.sql(
            modified_record_sql,
            result=result_df,
        ).collect()[0]["count"]

        logger.info(
            f"Finding modified records done, found '{modified_record_count}' modified records."
        )

        deleted_record_sql = f"""
            select count(*) as count
            from {{result}}
            where _reason = 'Delete'
                and _effective_to = '{current_datetime}'
        """

        logger.debug(
            f"Sql query to find deleted records: {deleted_record_sql}")
        logger.info("Finding deleted records...")

        deleted_record_count = spark_session.sql(
            deleted_record_sql,
            result=result_df,
        ).collect()[0]["count"]

        logger.info(
            f"Finding deleted records done, found '{deleted_record_count}' deleted records."
        )

        logger.info(f"Total target records: {result_df.count()}")

        logger.info("Collecting analytics done.")

    return result_df
