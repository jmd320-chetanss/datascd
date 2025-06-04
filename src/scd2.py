from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
from datetime import datetime
import logging

_empty_logger = logging.getLogger("my_empty_logger")
_empty_logger.addHandler(logging.NullHandler())


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

    assert _are_keys_valid(
        df=source_df, key_cols=key_cols
    ), "Key columns represent duplicate records."

    special_cols = ["_effective_from", "_effective_to", "_reason", "_active"]
    all_cols = [col for col in source_df.columns if col not in special_cols]
    non_key_cols = [col for col in all_cols if col not in key_cols]
    current_datetime_string = f"'{current_datetime}'"

    logger.debug(f"all_cols: {all_cols}")
    logger.debug(f"key_cols: {key_cols}")
    logger.debug(f"non_key_cols: {non_key_cols}")
    logger.debug(f"special_cols: {special_cols}")
    logger.debug(f"current_datetime: {current_datetime}")

    if target_df is None:
        target_df = spark_session.sql(
            f"""
                select
                    { ", ".join(all_cols) }
                    , null as _effective_from
                    , null as _effective_to
                    , null as _reason
                    , null as _active
                from {{source}}
                where false
            """,
            source=source_df,
        )

    # Sql query to generate target table with all the new records added,
    # updated records modified and added and deleted records marked deleted
    scd_sql = f"""
        with source as (
            select { ", ".join(all_cols) } from {{in_source}}
        ),
        target as (
            select * from {{in_target}}
        ),

        -- Records in source that are not present in target
        source_new as (
            select source.*
            from source
            left join target on { " and ".join([f"source.{col} = target.{col}" for col in key_cols]) }
            where { " and ".join([f"target.{col} is null" for col in key_cols]) }
        ),

        -- Records in source that are present in target and have been modified
        source_modified as (
            select source.*
            from source
            inner join target on { " and ".join([f"source.{col} = target.{col}" for col in key_cols]) }
            -- compare against the active records only
            where target._active is true
                {
                    (
                        " and ("
                        + " or ".join([f"source.{col} != target.{col}" for col in non_key_cols])
                        + ")"
                    ) if non_key_cols else ""
                }
        ),

        -- Records in target that are not present in source
        target_delete as (
            select target.*
            from target
            left join source on { " and ".join([f"source.{col} = target.{col}" for col in key_cols]) }
            -- Compare against the active records only
            where target._active is true
                and ({ " and ".join([f"source.{col} is null" for col in key_cols]) })
        )

        -- Pick all the new records from source and mark them as new
        select
            *
            , {current_datetime_string} as _effective_from
            , null as _effective_to
            , 'New' as _reason
            , true as _active
        from source_new

        union all

        -- Pick all the records from target that have been deleted and mark them as deleted
        select
            { ", ".join(f"target.{col}" for col in all_cols) }
            , target._effective_from
            , {current_datetime_string} as _effective_to
            , 'Delete' as _reason
            , false as _active
        from target_delete
        inner join target on { " and ".join([f"target.{col} = target_delete.{col}" for col in key_cols]) }

        union all

        -- Pick all the records from target that have been modified and mark them as deleted
        select
            { ", ".join(f"target.{col}" for col in all_cols) }
            , target._effective_from

            -- if the old record has been deleted and a new record comes with the same id,
            -- dont't overwrite the end date and reason of the old record
            , coalesce(target._effective_to, {current_datetime_string}) as _effective_to
            , case when target._active is false then target._reason else 'Update' end as _reason

            , false as _active
        from source_modified
        inner join target on { " and ".join([f"target.{col} = source_modified.{col}" for col in key_cols]) }

        union all

        -- Pick all the records from source that have been modified and mark them as updated
        select
            *
            , {current_datetime_string} as _effective_from
            , null as _effective_to
            , 'Update' as _reason
            , true as _active
        from source_modified

        union all

        -- Pick all the remaining (old, unmodified and active) records from target
        select target.*
        from target
        left join source_modified on { " and ".join([f"target.{col} = source_modified.{col}" for col in key_cols]) }
        left join target_delete on { " and ".join([f"target.{col} = target_delete.{col}" for col in key_cols]) }
        where { " and ".join([f"source_modified.{col} is null" for col in key_cols]) }
            and { " and ".join([f"target_delete.{col} is null" for col in key_cols]) }
    """

    logger.trace(f"Sql query to generate SCD2 target: {scd_sql}")
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
                and _effective_from = {current_datetime_string}
        """

        logger.trace(f"Sql query to find new records: {new_record_sql}")
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
                and _effective_from = {current_datetime_string}
        """

        logger.trace(f"Sql query to find modified records: {modified_record_sql}")
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
                and _effective_to = {current_datetime_string}
        """

        logger.trace(f"Sql query to find deleted records: {deleted_record_sql}")
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
