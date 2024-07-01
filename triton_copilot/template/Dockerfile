FROM nvcr.io/nvidia/tritonserver:##tritonversion##

ADD . /model/##model_name##

RUN pip install --upgrade pip

RUN pip install -r /model/##model_name##/1/requirements.txt

ENTRYPOINT tritonserver --model-repository /model