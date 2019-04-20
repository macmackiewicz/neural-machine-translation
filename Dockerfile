FROM python:3.5.2

COPY ./requirements.txt /opt/program/
WORKDIR /opt/program

RUN pip install -r requirements.txt

COPY ./nmt_cli.py ./setup.py ./train /opt/program/
COPY ./nmt/ /opt/program/nmt/
COPY ./scripts/ /opt/program/scripts/
RUN python setup.py install

ENV PATH="/opt/program:${PATH}"
