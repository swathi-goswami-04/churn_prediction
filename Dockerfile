FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8000
WORKDIR /app


COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


RUN mkdir -p /app/artifacts
COPY artifacts/preprocessor.joblib /app/artifacts/
COPY artifacts/best_model.joblib /app/artifacts/

COPY app.py /app/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]