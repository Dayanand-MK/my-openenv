FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY environment.py .
COPY email_data.py .
COPY tasks.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

EXPOSE 7860

CMD ["python", "server.py"]
