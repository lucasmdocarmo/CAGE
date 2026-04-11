FROM python:3.12-slim
WORKDIR /app

# Only install dependencies required to run the router.
# (Avoids heavyweight ML deps like torch/transformers.)
COPY docker/router.requirements.txt ./router.requirements.txt
RUN pip install --no-cache-dir -r router.requirements.txt

COPY src ./src

CMD ["python", "-m", "src.orchestration.router"]
EXPOSE 9000
