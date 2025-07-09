# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Set PYTHONPATH so that `src` package is discoverable
ENV PYTHONPATH=/app

# Install GNU Make without recommended extras to keep image slim
RUN apt-get update && apt-get install -y make \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Expose the port Streamlit runs on
EXPOSE 8501

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
