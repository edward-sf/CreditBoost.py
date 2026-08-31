# CreditBoost.py
A machine learning pipeline and API that predicts loan default risk for thin-file borrowers using alternative credit data.

**Domains:** `Lending`, `Risk Analytics`

## Tech Stack

- **Python (XGBoost, Scikit-Learn)**
- **FastAPI**
- **GitHub Actions**
- **Docker**

## Architecture

Trains an XGBoost gradient boosting model on public credit risk datasets. This project focuses heavily on the MLOPs pipeline: configuring GitHub Actions to automatically run unit tests on the model's inference logic, build a Docker container, and push the updated image to a registry.
