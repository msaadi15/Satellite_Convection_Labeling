FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps -e .
# Model weights are NOT baked into the image: mount them at /app/model
ENV CONVSEG_CONFIG=/app/configs/default.yaml
EXPOSE 8000
CMD ["convseg", "serve"]
