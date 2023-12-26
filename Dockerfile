FROM ubuntu:20.04


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    python3-pip \
    make \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

RUN apt-get update && apt-get install -y libncurses5 && rm -rf /var/lib/apt/lists/*

WORKDIR /barcode_service

COPY requirements.txt requirements.txt

RUN pip --no-cache-dir install -r  requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY . /barcode_service/

EXPOSE 5000

CMD make run_app

