from pyspark.sql import SparkSession
from pyspark.sql import types as spt
from collections import Counter
from datetime import datetime
import pytest
import src as datascd


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[*]").appName("pytest-spark").getOrCreate()
    )

    yield spark

    spark.stop()


def test_scd2(spark):

    timestamp = datetime.now()

    key_cols = ["id"]

    # Prepare the source DataFrame
    source_schema = spt.StructType(
        [
            spt.StructField("id", spt.IntegerType(), nullable=False),
            spt.StructField("name", spt.StringType(), nullable=True),
        ]
    )

    source_columns = source_schema.fieldNames()

    source_df = spark.createDataFrame(
        [
            (1, "Alice"),
            (2, "Bob"),
        ],
        source_schema,
    )

    # Prepare the expected Dataframe
    expected_schema = spt.StructType(
        [
            spt.StructField("id", spt.IntegerType(), nullable=False),
            spt.StructField("name", spt.StringType(), nullable=True),
            spt.StructField("_effective_from", spt.DateType(), nullable=False),
            spt.StructField("_effective_to", spt.DateType(), nullable=True),
            spt.StructField("_reason", spt.StringType(), nullable=False),
            spt.StructField("_active", spt.BooleanType(), nullable=False),
        ]
    )

    expected_columns = expected_schema.fieldNames()

    expected_df = spark.createDataFrame(
        [
            (1, "Alice", timestamp, None, "New", True),
            (2, "Bob", timestamp, None, "New", True),
        ],
        expected_schema,
    )

    # --------------------------------------------------------------------------
    # Perform the test
    # --------------------------------------------------------------------------

    result_df = datascd.scd2.generate(
        spark_session=spark,
        source_df=source_df,
        target_df=None,
        key_cols=key_cols,
        current_datetime=timestamp,
    )

    assert Counter(result_df.columns) == Counter(expected_columns)

    assert Counter(result_df.collect()) == Counter(expected_df.collect())
