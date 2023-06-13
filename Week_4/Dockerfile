FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app

RUN pip install pandas scikit-learn pyarrow requests  

COPY hw_4.py /app/hw_4.py

CMD ["python3", "/app/hw_4.py", "--year", "2022", "--month", "04"]