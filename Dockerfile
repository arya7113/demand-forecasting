FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit==1.50.0 pandas pyarrow
COPY app/ .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
