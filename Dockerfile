FROM ubuntu:22.04

# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip

# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY run.sh .
COPY train.py .

CMD ["bash","-c", "./app/run.sh"]
