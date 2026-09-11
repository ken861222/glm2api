# Runtime-only: glm2api is pure Python stdlib (dependencies = [] in pyproject).
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN useradd -u 10001 -m app

COPY src/ /app/src/
COPY main.py /app/main.py
COPY .env.example /app/.env.example

RUN mkdir -p /app/log && chown -R app:app /app

USER app

ENV PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

ENTRYPOINT ["python", "/app/main.py"]
