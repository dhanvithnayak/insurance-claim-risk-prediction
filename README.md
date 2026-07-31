<div id="top"></div>

<div align="center">
  <h1>Insurance Claim Risk Prediction</h1>
  <p>
    <i>Actuarially sound machine learning pipeline, containerized inference microservice, and interactive web dashboard for personal auto insurance risk assessment.</i>
    <br />
    <a href="https://github.com/dhanvithnayak/insurance-claim-risk-prediction/issues">Request Feature / Report Bug</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
<summary>Table of Contents</summary>

- [About The Project](#about-the-project)
  - [Key Performance Highlights](#key-performance-highlights)
  - [Cross-Validation Benchmark (5-Fold Stratified)](#cross-validation-benchmark-5-fold-stratified)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Local Setup](#installation--local-setup)
- [Usage](#usage)
  - [Interactive Web Dashboard](#interactive-web-dashboard)
  - [REST API Specification](#rest-api-specification)
  - [Docker Execution](#docker-execution)
- [Statistical Data Quality & Feature Engineering](#statistical-data-quality--feature-engineering)
- [Interpretability (SHAP Analysis)](#interpretability-shap-analysis)
- [Project Structure](#project-structure)
- [License](#license)

</details>

<!-- ABOUT THE PROJECT -->
## About The Project

In property and casualty auto insurance underwriting, accurately estimating the probability of a policyholder filing a claim is essential for risk-based pricing, solvency margin compliance, and loss-ratio optimization. Claims in personal auto policies are moderately imbalanced (in this benchmark dataset, only 31.33% of policyholders file a claim). Under standard accuracy metrics, a naive majority-class classifier achieves 68.67% accuracy while failing to identify a single high-risk claimant.

In insurance operations, failing to identify an applicant who files a claim (a False Negative) is far costlier than over-flagging a safe driver (a False Positive). Consequently, this project evaluates models primarily using **Precision-Recall AUC (PR-AUC)** and **Claim Recall**, ensuring that underwriting decisions prioritize loss prevention.

This repository provides an end-to-end production solution:
1. **Actuarial Modeling Pipeline:** Hypothesis-driven statistical audits ($\chi^2$ tests of missingness mechanisms, monotonic encoding validation), leakage-free preprocessing pipelines, and Stratified 5-Fold cross-validation comparing Balanced Logistic Regression, Random Forest, and XGBoost.
2. **Model Explainability:** SHAP (SHapley Additive exPlanations) attribution to extract risk drivers and inspect localized territory pricing anomalies.
3. **Inference Microservice:** High-throughput FastAPI REST API with Pydantic schema validation, automated median imputation for missing applicant attributes, and zero-downtime health probes.
4. **Interactive Web Dashboard:** Modern HTML5/CSS3 frontend with real-time risk gauges and interactive policy scoring.
5. **Containerization & CI/CD:** Hardened non-root multi-stage Docker container orchestrated via Docker Compose and validated on every commit with GitHub Actions.

<p align="right">(<a href="#top">back to top</a>)</p>

### Key Performance Highlights

- **PR-AUC**: **`0.8361 ± 0.0112`** (+166.9% relative lift over the **`0.3133`** random-guessing baseline).
- **Claim Recall**: **`86.69% ± 0.0112`** (captures nearly 9 out of 10 claimants, minimizing unpriced underwriting loss).
- **Cross-Validated Accuracy**: **`84.01% ± 0.0048`** (+15.34% over the 68.67% naive majority-class baseline).
- **Dominant Risk Factor**: `driving_experience` is the #1 protective driver (Mean |SHAP| = 1.66, coef = -1.96).

<p align="right">(<a href="#top">back to top</a>)</p>

### Cross-Validation Benchmark (5-Fold Stratified)

All candidate models were evaluated with balanced class weighting across identical stratified cross-validation folds with leakage-free preprocessing:

| Model | Accuracy (vs 68.67%) | Precision (Claim=1) | Recall (Claim=1) | F1-Score (Claim=1) | PR-AUC (vs 0.3133) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced) [WINNER]** | **0.8401 ± 0.0048** | 0.6969 ± 0.0082 | **0.8669 ± 0.0112** | **0.7726 ± 0.0061** | **0.8361 ± 0.0112** |
| Random Forest (Balanced) | 0.8328 ± 0.0064 | **0.7110 ± 0.0106** | 0.7858 ± 0.0091 | 0.7465 ± 0.0090 | 0.7901 ± 0.0101 |
| XGBoost (Weighted) | 0.8330 ± 0.0049 | 0.7036 ± 0.0081 | 0.8072 ± 0.0131 | 0.7518 ± 0.0074 | 0.8087 ± 0.0070 |

*Model Selection Defense:* While tree-based ensembles are popular, the true underlying risk landscape across driver experience and vehicle ownership is strictly monotonic. Calibrated, regularized logistic regression produces well-calibrated posterior probabilities without the step-function variance of decision trees on low-cardinality discrete variables.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow the instructions below to run the prediction service and web dashboard locally.

### Prerequisites

Ensure you have either Docker installed or Python 3.12+ available locally:
* **Docker & Docker Compose** (recommended):
  ```bash
  docker --version
  docker compose version
  ```
* **Python 3.12+**:
  ```bash
  python3 --version
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation & Local Setup

#### Option A: Docker Compose (Fastest)

Clone the repository and build the container:
```bash
git clone https://github.com/dhanvithnayak/insurance-claim-risk-prediction.git
cd insurance-claim-risk-prediction
docker compose up --build
```
The application will be live at `http://localhost:8000/`.

#### Option B: Local Python Virtual Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/dhanvithnayak/insurance-claim-risk-prediction.git
   cd insurance-claim-risk-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install pinned dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install pytest ruff
   ```

4. Run the end-to-end modeling pipeline (optional, pre-serialized artifact is included):
   ```bash
   python car_insurance_analysis.py
   ```

5. Run automated test suite:
   ```bash
   pytest -v
   ```

6. Launch the FastAPI server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Option C: Serverless Cloud Deployment (Vercel)

Deploy instantly with zero configuration and no credit card required:
1. Log in to [Vercel](https://vercel.com) using your GitHub account.
2. Click **Add New...** -> **Project**, and select `insurance-claim-risk-prediction`.
3. Click **Deploy**. Vercel will automatically detect `vercel.json` and `api/index.py`.
4. Your interactive web dashboard and FastAPI REST endpoints will be live at `https://insurance-claim-risk-prediction.vercel.app`.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Interactive Web Dashboard

Open `http://localhost:8000/` in your browser. The embedded dashboard provides:
- Form fields for driver demographics, vehicle properties, driving history, and territorial ZIP code.
- Checkboxes to simulate missing `credit_score` or `annual_mileage` fields, verifying pipeline imputation in real time.
- Dynamic SVG gauge displaying predicted claim probability, risk classification tier (Low / Moderate / High), and audit tags for imputed features.

### REST API Specification

The FastAPI microservice provides automated Swagger documentation at `http://localhost:8000/docs` and OpenAPI schema at `http://localhost:8000/openapi.json`.

#### `POST /predict`
Evaluates policyholder risk and returns calibrated probability and tier.

**Sample cURL Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 2,
    "gender": 1,
    "driving_experience": "20-29y",
    "education": "university",
    "income": "upper class",
    "credit_score": 0.629,
    "vehicle_ownership": 1.0,
    "vehicle_year": "after 2015",
    "married": 1.0,
    "children": 1.0,
    "postal_code": 10238,
    "annual_mileage": 12000.0,
    "vehicle_type": "sedan",
    "speeding_violations": 0,
    "duis": 0,
    "past_accidents": 0
  }'
```

**Sample JSON Response:**
```json
{
  "claim_probability": 0.0069,
  "predicted_claim": 0,
  "risk_tier": "Low",
  "missing_fields_imputed": []
}
```

*Fault-Tolerant Imputation:* If `credit_score` or `annual_mileage` is set to `null` or omitted, the model substitutes the training fold median and lists the affected fields inside `missing_fields_imputed` for underwriting auditability.

#### `GET /health`
Liveness and readiness probe for container orchestrators (Docker, Kubernetes).

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "service": "insurance-claim-risk-prediction",
  "model_loaded": true
}
```

### Docker Execution

To run in detached production mode:
```bash
docker compose up -d
```
To stop the service:
```bash
docker compose down
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- STATISTICAL DATA QUALITY -->
## Statistical Data Quality & Feature Engineering

1. **Missingness Mechanism ($\chi^2$ Test of Independence):**
   - In raw data, `credit_score` (9.8% missing) and `annual_mileage` (9.6% missing) contain nulls.
   - Contingency table tests against claims yielded $p = 0.8189$ (`credit_score`) and $p = 0.0966$ (`annual_mileage`).
   - Because missingness is unassociated with claim outcomes, the data satisfies Missing Completely At Random (MCAR) / Missing At Random (MAR). Dedicated missing-indicator columns were omitted to prevent overfitting, and median imputation was implemented within cross-validation folds.

2. **Monotonic Risk Decay & Ordinal Encoding:**
   - Analysis of historical claim frequency demonstrated strictly monotonic drops across socio-economic categories:
     - `income`: poverty (65.38%) $\to$ working class (45.33%) $\to$ middle class (27.69%) $\to$ upper class (13.35%).
     - `education`: none (47.15%) $\to$ high school (32.33%) $\to$ university (22.56%).
   - Ordinal encoding was selected over one-hot encoding to preserve this natural ordering using a single degree of freedom, preventing dimensionality inflation and linear collinearity.

3. **Postal Code Territory Audit:**
   - The dataset contains 4 discrete postal codes (10238, 21217, 32765, 92101). Storing postal codes as raw integers creates an erroneous continuous spatial topology.
   - Categorical one-hot encoding isolates territorial risk. Notably, ZIP code `21217` exhibited a 100.0% claim rate ($N = 120$), receiving a high positive log-odds weight (+7.68) indicative of a localized risk cluster.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- INTERPRETABILITY -->
## Interpretability (SHAP Analysis)

Global feature attributions were computed using SHAP values on the winning model:

| Rank | Feature | Mean \|SHAP\| | Model Coef | Actuarial Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `driving_experience` | **1.6586** | -1.9625 | Dominant protective driver; experience dramatically lowers claim propensity. |
| **2** | `vehicle_ownership` | **0.8317** | -1.8534 | Owning vehicle reduces claim likelihood vs leasing/borrowing. |
| **3** | `vehicle_year_before 2015` | **0.7549** | +1.8981 | Older vehicles lack modern ADAS safety systems and exhibit higher claim frequency. |
| **4** | `gender` (Male) | **0.5202** | +1.0406 | Male drivers exhibit statistically higher claim rates in this demographic. |
| **5** | `postal_code_32765` | **0.4361** | +1.2742 | Suburban geographic territory risk factor. |
| **6** | `annual_mileage` | **0.2638** | +0.3417 | Exposure risk: increased vehicle miles traveled linearly increases accident probability. |
| **7** | `postal_code_21217` | **0.2420** | +7.6765 | Highly concentrated risk pocket (100% historical claim rate). |
| **8** | `married` | **0.1887** | -0.3775 | Married status corresponds to lower risk profiles. |

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- PROJECT STRUCTURE -->
## Project Structure

```text
insurance-claim-risk-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated CI/CD (Lint, Pytest, Docker Smoke Test)
├── data/
│   └── car_insurance.csv       # 10,000 policyholder records
├── templates/
│   └── index.html              # HTML5/CSS3 interactive risk assessment dashboard
├── tests/
│   └── test_api.py             # 6 automated regression and integration test cases
├── .dockerignore               # Container build exclusion patterns
├── .gitignore                  # Git tracking rules (caches, virtualenvs, bytecode)
├── app.py                      # Production FastAPI inference microservice
├── api/                        # Vercel serverless deployment entrypoint & requirements
│   ├── index.py
│   └── requirements.txt
├── car_insurance_analysis.py   # Statistical audit, cross-validation & SHAP pipeline
├── Dockerfile                  # Hardened, non-privileged Python 3.12 container
├── docker-compose.yml          # Container orchestration configuration
├── LICENSE                     # MIT License
├── model_pipeline.joblib       # Serialized Scikit-Learn production artifact
├── pytest.ini                  # Pytest configuration settings
├── requirements.txt            # Pinned dependency specifications
└── vercel.json                 # Vercel serverless function routing configuration
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.

<p align="right">(<a href="#top">back to top</a>)</p>
