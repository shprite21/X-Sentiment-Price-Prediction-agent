FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY tests ./tests
ENTRYPOINT ["python", "-m", "xsentiment"]
CMD ["demo"]
