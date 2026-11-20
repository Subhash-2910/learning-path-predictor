# Learning Path Predictor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/9nsnx7z2dn/badge.svg)](https://ask.thili.ai/b/9nsnx7z2dn)

This project, 'Learning Path Predictor,' is an EdTech application designed to optimize student learning paths. It recommends relevant next courses or study materials by leveraging a language model trained on sequences of completed courses, much like predicting items in a shopping basket.

## What you are building

**Goal.** You will build a Learning Path Predictor that recommends next courses or study materials to students based on their completed courses.

**Done when.** The project is complete when the model successfully suggests relevant next courses, proven by displaying the top N recommendations for a user's chosen completed courses in the Streamlit interface.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Generate mock student course completion data as sequences of course IDs.**  
      Hint: Generate synthetic data for course sequences, similar to mock retail transactions.
- [x] **2. Pretrain a small language model on these course ID sequences.**  
      Hint: Apply language model fine-tuning techniques from the drill to these course ID sequences.
- [ ] **3. Implement a function to predict the next likely course given a partial sequence.**  
      Hint: Use the trained model's prediction capabilities to forecast the next course ID.
- [ ] **4. Create a Streamlit interface for students to input their completed courses and view recommendations.**  
      Hint: Utilize Streamlit to build an interactive web application for user input and recommendation display.
- [ ] **5. Analyze which course "itemsets" the model has learned.**  
      Hint: Examine the model's learned relationships between courses, possibly through embedding analysis or attention weights.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow students to provide feedback on recommendations (e.g., 'not interested') to refine future suggestions.
- Display the model's confidence score or probability alongside each recommended course.
- Implement a feature to explore courses similar to a recommended one, using embedding similarity.
