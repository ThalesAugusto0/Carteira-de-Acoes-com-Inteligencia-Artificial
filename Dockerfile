FROM python:3
COPY . /app  
RUN apt update && apt install iproute2 -y
RUN pip install pandas && pip install -U scikit-learn && pip install plotly && pip install -U matplotlib && pip install seaborn && pip install ipython && pip install xlrd && pip install openpyxl && pip install pandas_datareader
WORKDIR /app
CMD python ia.py