from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import types as spt
from collections import Counter
from datetime import datetime
import pytest
import src as datascd


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master(
            "local[*]").appName("pytest-spark").getOrCreate()
    )

    yield spark

    spark.stop()


def perform_scd_test(
    spark,
    key_cols: list[str],
    source_df: DataFrame,
    target_df: DataFrame | None,
    expected_df: DataFrame,
    timestamp: datetime = None,
):

    result_df = datascd.scd2.generate(
        spark_session=spark,
        source_df=source_df,
        target_df=target_df,
        key_cols=key_cols,
        current_datetime=timestamp,
    )

    assert Counter(result_df.columns) == Counter(expected_df.columns)
    assert Counter(result_df.collect()) == Counter(expected_df.collect())


def test_scd2(spark):

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
                spt.StructField("_effective_from", spt.TimestampType(), nullable=False),
                spt.StructField("_effective_to", spt.TimestampType(), nullable=True),
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
    )
