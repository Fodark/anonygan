FROM nvcr.io/nvidia/pytorch:21.05-py3
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /src