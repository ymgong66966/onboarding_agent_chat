FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY chat_server.py .
COPY chat_ui/ chat_ui/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["python", "chat_server.py", "--host", "0.0.0.0", "--port", "8001"]
