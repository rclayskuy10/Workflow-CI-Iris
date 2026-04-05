FROM python:3.12.7-slim

WORKDIR /app

COPY requirements_docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "runs:/47fc0ee9002145a3b059f85a61eca2bf/model", "--host", "0.0.0.0", "--port", "5000", "--no-conda"]
