FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV FLASK_ENV=production
CMD ["python", "app.py"]