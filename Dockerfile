FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

RUN pip install \
batchgenerators==0.24 \
certifi==2022.6.15.1 \
charset-normalizer==2.1.1 \
future==0.18.2 \
idna==3.3 \
imageio==2.33 \
joblib==1.4.2 \
linecache2==1.0.0 \
networkx==2.8.6 \
nibabel==4.0.2 \
numpy==1.26.3 \
opencv-python==4.6.0.66 \
packaging==24.1 \
pandas==2.2.3 \
Pillow==9.2.0 \
pyparsing==3.0.9 \
python-dateutil==2.8.2 \
python-dotenv==0.21.0 \
pytz==2022.2.1 \
PyWavelets==1.7.0 \
requests==2.28.1 \
scikit-image==0.24.0 \
scipy==1.14.1 \
six==1.16.0 \
threadpoolctl==3.1.0 \
tifffile==2022.8.12 \
traceback2==1.4.0 \
typing-extensions==4.12.2 \
unittest2==1.1.0 \
urllib3==1.26.12

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


COPY app/ /app/

WORKDIR /app/
