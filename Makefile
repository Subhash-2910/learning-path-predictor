# One command to rebuild everything: make all
PY ?= .venv/bin/python

.PHONY: all data train evaluate analyze app test docker
all: data train evaluate analyze

data:      ## Milestone 1: 10,000 synthetic students -> dataset/
	$(PY) -m app.data
train:     ## Milestone 2: pretrain the course GPT -> model/course_gpt.pt, results/training.json
	$(PY) -m app.model
evaluate:  ## Milestone 3: next-course prediction vs baselines -> results/metrics.md, results/samples.md
	$(PY) -m app.predict
analyze:   ## Milestone 5: learned itemsets, embeddings, attention -> results/itemsets.md, *.png
	$(PY) -m app.analyze
app:       ## Milestone 4: the Streamlit app on http://localhost:8501
	$(PY) -m streamlit run app/main.py
test:
	$(PY) -m pytest -q
docker:    ## Build and run the app in Docker on http://localhost:8080
	docker build -t learning-path-predictor . && docker run --rm -p 8080:8080 learning-path-predictor
