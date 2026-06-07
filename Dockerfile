FROM python:3.11-slim

WORKDIR /app

# Copy source first, then install (setuptools needs src/ for dynamic version)
COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "agentcore", "serve", "--host", "0.0.0.0", "--port", "8000"]
