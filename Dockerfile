FROM python:3.6-slim-buster
RUN useradd --create-home recommender
WORKDIR /home/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN chown -R recommender:recommender /home/app
USER recommender
CMD ["python","-u","main.py"]