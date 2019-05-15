FROM python:3.5.2

COPY ./requirements.txt ./requirements-dev.txt ./constraints.txt /opt/program/
WORKDIR /opt/program

RUN pip install -r requirements-dev.txt

COPY ./train ./nmt_cli.py ./setup.py /opt/program/
COPY ./nmt/ /opt/program/nmt/
COPY ./scripts/ /opt/program/scripts/
COPY ./cloud_runner/ /opt/program/cloud_runner/

RUN python setup.py install

ENTRYPOINT ["nmt"]
