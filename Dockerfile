# ==============================================================================
# FinGuard: Credit Card Fraud Detection Platform Dockerfile
# Base: Python 3.11 Slim with OpenJDK 17 for Apache PySpark MLlib
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OpenJDK 17 and core Linux utilities required by Apache Spark
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Java and Spark environment variables
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON="python3"
ENV PYSPARK_DRIVER_PYTHON="python3"

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files, models, and datasets
COPY . .

# Expose default Streamlit web port
EXPOSE 8501

# Container Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app binding to all network interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
