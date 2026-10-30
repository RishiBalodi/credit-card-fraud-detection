"""
Spark Session Management Module
===============================
Provides singleton SparkSession initialization with Windows-compatible
network configuration and Python 3.11 interpreter bindings.
"""

import os
import sys

# Configure environment variables to ensure PySpark worker & driver
# use the correct Python 3.11 interpreter on Windows
PYTHON_311 = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
if os.path.exists(PYTHON_311):
    os.environ["PYSPARK_PYTHON"] = PYTHON_311
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_311
else:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

_spark_instance = None


def get_spark_session(app_name: str = "CreditCardFraudDetection", master: str = "local[2]") -> SparkSession:
    """
    Creates or retrieves an active PySpark SparkSession.
    Configured specifically for Windows local execution with loopback IP binding.
    """
    global _spark_instance
    if _spark_instance is not None:
        try:
            if not _spark_instance._jsc.sc().isStopped():
                return _spark_instance
        except Exception:
            _spark_instance = None

    _spark_instance = (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    # Set log level to WARN to eliminate verbose Spark terminal output
    _spark_instance.sparkContext.setLogLevel("WARN")
    return _spark_instance


def stop_spark_session():
    """
    Safely stops and cleans up the active SparkSession instance.
    """
    global _spark_instance
    if _spark_instance is not None:
        try:
            _spark_instance.stop()
        except Exception:
            pass
        finally:
            _spark_instance = None
