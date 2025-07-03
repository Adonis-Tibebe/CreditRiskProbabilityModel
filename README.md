# CreditRiskPredictionModel

# Credit Scoring Business Understanding

## Basel II Overview

**Basel II** is an international banking regulation framework created by the Basel Committee on Banking Supervision (BCBS). It was introduced to strengthen how banks measure and manage **credit risk**, **market risk**, and **operational risk** — the main risks that threaten a bank’s capital and stability.

### Risk-Sensitive Capital Requirements

Basel II has **three pillars:**

**Pillar 1 – Minimum Capital Requirements**  
- Banks must measure risks (credit, market, operational) and hold enough capital to cover unexpected losses.  
- Basel II introduced the Internal Ratings-Based (IRB) Approach, allowing banks to build internal models to estimate **Probability of Default (PD)**, **Loss Given Default (LGD)**, **Exposure at Default (EAD)**, and **Effective Maturity (M)**.

**Pillar 2 – Supervisory Review**  
- Banks must have sound risk management processes.  
- Regulators review banks’ models to ensure they are robust, reasonable, and well-documented.

**Pillar 3 – Market Discipline**  
- Banks must disclose their risk measurement methods and risk exposures.  
- Transparency ensures external stakeholders can assess a bank’s soundness.

### Emphasis on Risk Measurement

- **More granular measurement:** Instead of rough averages, banks must measure individual exposures precisely.
- **Internal models:** Banks can use internal statistical models — but they must be accurate, validated, and transparent.
- **Data-driven & explainable:** Supervisors expect clear documentation, logic, and ongoing performance checks. No “black box” models: banks must explain how risk estimates are calculated.

**Basel II’s focus on risk measurement means our credit score & risk probability model must be:**  
- **Accurate:** Truly reflect each customer’s default risk.  
- **Interpretable:** We can explain why a customer is high or low risk.  
- **Well-documented:** Every data source, feature, assumption, and result is clear and traceable.  
- **Monitored:** We regularly check if it works well and adjust as needed.

---

## Need for a Proxy to Measure Default

In traditional lending, “default” means the borrower legally fails to repay according to contract. In our data, we do not have direct measures of default (e.g., **PaymentStatus**, **DueDate**, **PaidDate**, **OutstandingBalance**).  

So we must approximate default risk using **proxy behaviors**, for example:  
- Late payments (30/60/90+ days late).  
- Frequent partial payments.  
- Chargebacks or disputes.  
- Users who never return after owing money.

These behaviors act as a **proxy** for true default.

### Business Risks of Using Proxies

- **Wrong Signal → Wrong Predictions:** If our proxy does not truly represent “people who fail to pay fully,” we might misclassify low-risk customers as high risk — or worse, approve high-risk customers. Example: A customer could be late because of a payment system glitch, not because they’re risky.

- **Hidden Bias:** Proxies might unintentionally reflect demographic or behavioral biases unrelated to creditworthiness. E.g., international customers may pay late due to currency delays.

- **Poor Model Generalization:** A narrow proxy may perform well on training data but fail in the real world when true defaults occur.

- **Regulatory & Reputation Risk:** Predicting risk with a bad proxy can lead to unfair or discriminatory decisions, causing complaints, regulatory scrutiny, or reputational harm.

**Therefore, when true default labels are unavailable, a proxy is necessary — but we must:**  
- Choose the best available proxy that signals non-repayment.  
- Clearly document and justify our choice.  
- Monitor the model as real default data builds up — refine or replace the proxy over time.

---

## Trade-Off: Simple vs. Complex Models

**Simple Model — Logistic Regression + WoE:**  
- Uses **Weight of Evidence (WoE)** to bin variables into risk groups, stabilizing noisy data and keeping relationships monotonic.  
- Logistic regression maps these bins to a **linear log-odds equation**; each coefficient shows how much that variable shifts default probability.  
- Easy to interpret: clear, auditable logic (e.g., higher income → lower risk).  
- Low overfitting risk if bins are well-designed.

**Complex Model — Gradient Boosting (e.g., XGBoost, LightGBM):**  
- Builds an **ensemble of decision trees**, each correcting errors from the last to capture **nonlinear patterns and interactions** automatically.  
- Usually higher predictive performance (higher AUC/accuracy).  
- But the final model is a **black box**; coefficients alone don’t explain decisions — requires post-hoc tools (e.g., **SHAP**, **LIME**) for interpretability.  
- Higher risk of hidden bias or drift if not strictly governed.

In regulated credit scoring, **logistic regression + WoE** is trusted because it is **linear, monotonic, and traceable**. Gradient boosting can yield extra accuracy but demands robust **explainability tools**, detailed documentation, and stronger validation to stay compliant with regulatory standards.

---

**In summary**, this balance ensures our BNPL credit scoring model remains transparent, explainable, and responsibly governed — aligned with the principles of Basel II.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Add usage instructions here.

---

## Project Structure

```
CreditRiskProbabilityModel/
│
├── src/              # Main source code (API, models, services, utils)
├── notebooks/        # Jupyter notebooks for EDA, modeling, and tests
├── tests/            # Unit and integration tests
├── data/             # Data files (tracked by DVC, not in git)
├── models/           # Model artifacts and MLflow runs
├── scripts/          # Utility scripts
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker build file
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
└── .dvc/             # DVC config and cache (after dvc init)
```

---
## How to Run

### Local (development server)
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### With Docker
```bash
docker-compose build
docker-compose up
```
Then visit [http://localhost:8000/docs](http://localhost:8000/docs) for the API docs.

---
## Testing

- **Lint:**  
  ```bash
  flake8 src/ --max-line-length=120
  ```
- **Unit & Integration Tests:**  
  ```bash
  pytest tests/
  ```
- **GitHub Actions:**  
  Tests and linting are run automatically on every push to the main branch.

---
## DVC Usage

- **Initialize DVC:**  
  ```bash
  dvc init
  ```
- **Track data:**  
  ```bash
  dvc add data/raw
  ```
- **Set up a local remote:**  
  ```bash
  dvc remote add -d localremote ../dvcstore
  ```
- **Push data to remote:**  
  ```bash
  dvc push
  ```
  ## Contributing

1. Fork the repo and create your branch from `main`.
2. Make your changes and add tests as needed.
3. Run `flake8` and `pytest` to ensure code quality.
4. Submit a pull request.

---

## License

Add your license information here.

---