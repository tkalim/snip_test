From pytorch/pytorch

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install numpy -r /requirements.txt

WORKDIR /code

CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/code
