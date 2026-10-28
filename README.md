# AI Medical Diagnosis Platform

A Flask-based web application for managing patient and diagnosis records and exploring AI-assisted clinical predictions. It combines a patient-management interface with tabular machine learning, symptom-text classification, and chest-image classification.

> **Important:** This project is for learning and research only. Its predictions and visualizations are not medical advice and must not be used to diagnose, treat, or rule out a condition. Any real-world clinical use would require appropriate clinical validation, regulatory review, privacy controls, and oversight by qualified healthcare professionals.

## Features

- Login-protected dashboard, patient records, diagnoses, clinical assessments, and reports.
- Symptom narrative classification using TF-IDF and Logistic Regression.
- Optional tabular clinical assessment using a saved Random Forest pipeline.
- Pneumonia image classification using an enhanced CNN, with a Grad-CAM visualization.
- AI model catalog and account settings.
- MySQL persistence through Flask-SQLAlchemy and Flask-Migrate.
- Jupyter notebooks and included datasets for the model experiments.

## Technology

- Python 3.10
- Flask, Flask-SQLAlchemy, Flask-Migrate, and MySQL
- pandas, scikit-learn, TensorFlow, Pillow, and scikit-image
- HTML, CSS, and JavaScript templates

## Project layout

```text
backend/       Flask routes, controllers, services, models, and utilities
database/      Alembic migration environment and schema revisions
datasets/      Datasets used by the experiments
models/        NLP and image-model artifacts; clinical ML model is generated locally
notebooks/     Model exploration and training notebooks
scripts/       Administrative and project utility scripts
static/        CSS, JavaScript, and icons
templates/     Jinja templates
tests/         Automated tests
app.py         Local Flask application entry point
config.py      Environment-based application and database configuration
```

## Requirements

- Python 3.10 and `pip`
- MySQL Server

## Local setup

Run commands from the project root.

### 1. Create a virtual environment and install dependencies

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Create the MySQL database

Create a database before applying the migrations. For example, from a MySQL client:

```sql
CREATE DATABASE ai_medical_diagnosis
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

### 3. Configure environment variables

Create a `.env` file in the project root. Set the values for your local MySQL account and generate a unique secret key; do not use the example placeholder as a real secret.

```dotenv
SECRET_KEY=replace-with-a-long-random-secret
LOG_LEVEL=INFO
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_NAME=ai_medical_diagnosis
DATABASE_USER=your_mysql_user
DATABASE_PASSWORD=your_mysql_password
```

Generate a secret key with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

The application loads `.env` automatically for local development. Deployment-provided environment variables take precedence.

### 4. Apply database migrations

The migration scripts are in `database/migrations`, so pass that directory explicitly:

```bash
flask --app app db -d database/migrations upgrade
```

### 5. Create an administrator and start the app

Create the first administrator after the schema has been migrated:

```bash
python scripts/create_admin.py
```

Then start the local development server:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser and sign in with the administrator account you created. `app.py` runs Flask in development mode; do not use Flask’s development server for a production deployment.

## Tests

Install pytest in the active environment if it is not already available:

```bash
python -m pip install pytest
python -m pytest
```

## AI artifacts and datasets

The NLP and image prediction features use serialized model files under `models/`. The clinical ML prediction feature also needs its Random Forest artifact, which is generated locally by the notebook below and is intentionally not stored in Git. The notebooks under `notebooks/` document the model experiments.

### Generating the Clinical ML Model

The trained Random Forest model is not included in this repository because of its large file size. Generate it before using the clinical ML prediction feature:

1. Install the dependencies listed in `requirements.txt` in your Python environment.
2. Install JupyterLab if it is not already available: `python -m pip install jupyterlab`.
3. Make sure the clinical dataset is available at `datasets/ml/chronic_disease/Data Warehouse Multiclass.csv`.
4. From the project root, launch Jupyter with `notebooks/` as its working directory:

   ```bash
   cd notebooks
   jupyter lab
   ```

5. Open `01_ML_Clinical_Disease_Prediction.ipynb` and run the notebook cells in order. The training and export cells save the fitted pipeline to `models/ml/clinical_disease_random_forest.joblib`.
6. Start the Flask application after the model has been generated.

The notebook trains a Random Forest with 200 estimators using the included clinical dataset. Training time and prediction performance depend on your system and the dataset. The dataset is approximately 89 MB; review its license and redistribution terms before publishing or sharing it.

## Before publishing to GitHub

- Keep `.env`, passwords, secret keys, and other credentials out of commits.
- Do not publish local patient information, uploaded images, generated Grad-CAM files, or other private data in `uploads/`, `instance/`, or `logs/`.
- Do not upload the local Python environment (`.venv/` or `aipp/`); recreate it from `requirements.txt`.
- The generated clinical Random Forest model is excluded from Git; recreate it by following [Generating the Clinical ML Model](#generating-the-clinical-ml-model).
- Review the dataset and model licenses and attribution requirements before redistributing them.
