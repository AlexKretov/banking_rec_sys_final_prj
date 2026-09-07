FROM python:3.11-slim
LABEL author=${AUTHOR}

#COPY . ./ml_service
COPY . ./home/
WORKDIR home/ml_service 

RUN pip3 install -r ../requirements.txt
VOLUME /metrics_reports
EXPOSE ${THE_PORT}
CMD uvicorn app2:app --reload --host ${THE_HOST} --port ${THE_PORT} 
