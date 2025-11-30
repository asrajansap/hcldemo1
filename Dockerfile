# python-gateway/Dockerfile

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1
 
COPY ./app /app/app

COPY requirements.txt /app/requirements.txt
 
RUN pip install --no-cache-dir -r /app/requirements.txt
 
# ensure data dir exists

RUN mkdir -p /app/data
 
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]

 
