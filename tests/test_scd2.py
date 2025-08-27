from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import types as spt
from collections import Counter
from datetime import datetime, timedelta
import logging
import pytest
import sys
import src as datascd


@pytest.fixture(scope="session")
def spark():

    spark = (
        SparkSession.builder.master(
            "local[*]").appName("pytest-spark").getOrCreate()
    )

    yield spark

    spark.stop()


@pytest.fixture(scope="session")
def logger():

    # Create logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    yield logger


def perform_scd_test(
    spark,
    key_cols: list[str],
    source_df: DataFrame,
    target_df: DataFrame | None,
    expected_df: DataFrame,
    logger: logging.Logger,
    timestamp: datetime = None,
) -> DataFrame:

    result_df = datascd.scd2.generate(
        spark_session=spark,
        source_df=source_df,
        target_df=target_df,
        key_cols=key_cols,
        current_datetime=timestamp,
        logger=logger,
        log_analytics=False,
    )

    assert Counter(result_df.columns) == Counter(expected_df.columns)
    assert Counter(result_df.collect()) == Counter(expected_df.collect())

    return result_df


# --------------------------------------------------------------------------------------------------
# Basic test case for SCD2
# --------------------------------------------------------------------------------------------------
def test_scd2_basic(spark: SparkSession, logger: logging.Logger):

    timestamp = datetime.now()

    # Prepare the source DataFrame
    source_df = spark.createDataFrame(
        [
            (1, "Alice"),
            (2, "Bob"),
        ],
        spt.StructType(
            [
                spt.StructField("id", spt.IntegerType(), nullable=False),
                spt.StructField("name", spt.StringType(), nullable=True),
            ]
        ),
    )

    # Prepare the expected Dataframe
    expected_df = spark.createDataFrame(
        [
            (1, "Alice", timestamp, None, "New", True),
            (2, "Bob", timestamp, None, "New", True),
        ],
        spt.StructType(
            [
                spt.StructField("id", spt.IntegerType(), nullable=False),
                spt.StructField("name", spt.StringType(), nullable=True),
                spt.StructField("_effective_from",
                                spt.TimestampType(), nullable=False),
                spt.StructField("_effective_to",
                                spt.TimestampType(), nullable=True),
                spt.StructField("_reason", spt.StringType(), nullable=False),
                spt.StructField("_active", spt.BooleanType(), nullable=False),
            ]
        ),
    )

    perform_scd_test(
        spark,
        key_cols=["id"],
        source_df=source_df,
        target_df=None,
        expected_df=expected_df,
        timestamp=timestamp,
        logger=logger,
    )

# --------------------------------------------------------------------------------------------------
# SCD2 Test Case: Schema change, new column in source
# --------------------------------------------------------------------------------------------------
def test_scd2_add_col(spark: SparkSession, logger: logging.Logger):

    timestamp_0 = datetime.now()
    timestamp_1 = timestamp_0 + timedelta(seconds=5)

    # Prepare the expected Dataframe
    source_df = spark.createDataFrame(
        [
            (1, "Alice", "India"),
            (2, "Bob", "India"),
        ],
        spt.StructType(
            [
                spt.StructField("id", spt.IntegerType(), nullable=False),
                spt.StructField("name", spt.StringType(), nullable=True),
                spt.StructField("country", spt.StringType(), nullable=True),
            ]
        ),
    )

    target_df = spark.createDataFrame(
        [
            (1, "Alice", timestamp_0, None, "New", True),
            (2, "Bob", timestamp_0, None, "New", True),
        ],
        spt.StructType(
            [
                spt.StructField("id", spt.IntegerType(), nullable=False),
                spt.StructField("name", spt.StringType(), nullable=True),
                spt.StructField("_effective_from",
                                spt.TimestampType(), nullable=False),
                spt.StructField("_effective_to",
                                spt.TimestampType(), nullable=True),
                spt.StructField("_reason", spt.StringType(), nullable=False),
                spt.StructField("_active", spt.BooleanType(), nullable=False),
            ]
        ),
    )

    # Prepare the expected Dataframe
    expected_df = spark.createDataFrame(
        [
            (1, "Alice", "India", timestamp_1, None, "Update", True),
            (1, "Alice", None, timestamp_0, timestamp_1, "Update", False),
            (2, "Bob", "India", timestamp_1, None, "Update", True),
            (2, "Bob", None, timestamp_0, timestamp_1, "Update", False),
        ],
        spt.StructType(
            [
                spt.StructField("id", spt.IntegerType(), nullable=False),
                spt.StructField("name", spt.StringType(), nullable=True),
                spt.StructField("country", spt.StringType(), nullable=True),
                spt.StructField("_effective_from",
                                spt.TimestampType(), nullable=False),
                spt.StructField("_effective_to",
                                spt.TimestampType(), nullable=True),
                spt.StructField("_reason", spt.StringType(), nullable=False),
                spt.StructField("_active", spt.BooleanType(), nullable=False),
            ]
        ),
    )

    perform_scd_test(
        spark,
        key_cols=["id"],
        source_df=source_df,
        target_df=target_df,
        expected_df=expected_df,
        timestamp=timestamp_1,
        logger=logger,
    )

# --------------------------------------------------------------------------------------------------
# SCD2 Test Case: Schema change, column removed from source
# --------------------------------------------------------------------------------------------------
def test_scd2_del_col(spark: SparkSession, logger: logging.Logger):

    timestamp_0 = datetime.now()
    timestamp_1 = timestamp_0 + timedelta(seconds=5)

    # Prepare the expected Dataframe
    source_df = spark.createDataFrame(
        [
            (1, "Alice"),
            (2, "Bob"),
        ],
        spt.StructType(
            [
                spt.StructField("id", spt.IntegerType(), nullable=False),
                spt.StructField("name", spt.StringType(), nullable=True),
            ]
        ),
    )

    target_df = spark.createDataFrame(
        [
            (1, "Alice", "India", timestamp_0, None, "New", True),
            (2, "Bob", "India", timestamp_0, None, "New", True),
        ],
        spt.StructType(
            [
                spt.StructField("id", spt.IntegerType(), nullable=False),
                spt.StructField("name", spt.StringType(), nullable=True),
                spt.StructField("country", spt.StringType(), nullable=True),
                spt.StructField("_effective_from",
                                spt.TimestampType(), nullable=False),
                spt.StructField("_effective_to",
                                spt.TimestampType(), nullable=True),
                spt.StructField("_reason", spt.StringType(), nullable=False),
                spt.StructField("_active", spt.BooleanType(), nullable=False),
            ]
        ),
    )

    # Prepare the expected Dataframe
    expected_df = spark.createDataFrame(
        [
            (1, "Alice", None, timestamp_1, None, "Update", True),
            (1, "Alice", "India", timestamp_0, timestamp_1, "Update", False),
            (2, "Bob", None, timestamp_1, None, "Update", True),
            (2, "Bob", "India", timestamp_0, timestamp_1, "Update", False),
        ],
        spt.StructType(
            [
                spt.StructField("id", spt.IntegerType(), nullable=False),
                spt.StructField("name", spt.StringType(), nullable=True),
                spt.StructField("country", spt.StringType(), nullable=True),
                spt.StructField("_effective_from",
                                spt.TimestampType(), nullable=False),
                spt.StructField("_effective_to",
                                spt.TimestampType(), nullable=True),
                spt.StructField("_reason", spt.StringType(), nullable=False),
                spt.StructField("_active", spt.BooleanType(), nullable=False),
            ]
        ),
    )

    perform_scd_test(
        spark,
        key_cols=["id"],
        source_df=source_df,
        target_df=target_df,
        expected_df=expected_df,
        timestamp=timestamp_1,
        logger=logger,
    )
