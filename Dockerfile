FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "safety_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]

