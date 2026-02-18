FROM amd64/python:3.9-buster
RUN pip install --upgrade pip
WORKDIR /app
COPY . /app
RUN pip --no-cache-dir install -r requirements.txt
WORKDIR /app/app
CMD ["python3", "run_server.py"]
