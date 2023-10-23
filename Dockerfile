FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install --upgrade pip

RUN cat requirements.txt

RUN apt-get update && apt-get install -y gcc libc6-dev

RUN pip install -r requirements.txt
CMD ["python3", "main.py"]