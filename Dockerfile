FROM pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel
 
WORKDIR /workspace
ENV PYTHONPATH=.

RUN apt update && \
    apt upgrade -y && \
    apt install -y xvfb

COPY . .
RUN pip install -r requirements.txt
