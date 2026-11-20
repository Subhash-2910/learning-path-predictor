FROM python:3.12-slim
# git is needed because thilimarketbasket installs straight from GitHub (see requirements.txt)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
# CPU-only PyTorch keeps the image small; the model is tiny and needs no GPU
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# The trained model, dataset and results are in the repo, so the image works without training
COPY . .
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "streamlit run app/main.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true"]
