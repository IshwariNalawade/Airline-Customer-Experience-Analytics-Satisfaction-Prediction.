# Airline Passenger Satisfaction Analytics & Machine Learning Pipeline

An end-to-end Data Science and Machine Learning project analyzing **129,880 passenger records** to uncover operational and in-flight drivers of customer satisfaction, benchmark classification models, and deploy an interactive analytics web application.

---

## Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Architecture & Cleaning](#-dataset-architecture--cleaning)
3. [Exploratory Data Analysis (EDA) & Insights](#-exploratory-data-analysis-eda--insights)
4. [Machine Learning Pipeline & Modeling](#-machine-learning-pipeline--modeling)
5. [Model Evaluation & Benchmarking](#-model-evaluation--benchmarking)
6. [Feature Importance & Interpretability](#-feature-importance--interpretability)
7. [Streamlit Interactive Web Application](#-streamlit-interactive-web-application)
8. [Repository Structure](#-repository-structure)
9. [Getting Started & Installation](#-getting-started--installation)
10. [Tech Stack](#️-tech-stack)

---

## Project Overview
In the aviation industry, passenger loyalty and customer experience (CX) directly drive profitability and brand equity. This project provides a comprehensive data-driven investigation into what makes airline passengers satisfied or dissatisfied. By combining rigorous statistical EDA, robust classification modeling (Random Forest achieving **95.9% accuracy**), and a deployed **Streamlit web dashboard**, this project bridges the gap between raw data analysis and actionable business intelligence.

---

## Dataset Architecture & Cleaning
* **Source:** Invistico Airline Passenger Satisfaction Dataset
* **Dimensions:** 129,880 rows × 23 columns (mixed numeric ratings, categorical features, and flight metrics).
* **Target Variable:** `satisfaction` (Binary: `satisfied` vs. `dissatisfied`).
* **Data Cleaning & Preprocessing:**
  * Handled missing values in `Arrival Delay in Minutes` (393 missing entries) by imputing via median departure delay dynamics without inducing data leakage.
  * Audited service rating columns (scaled 0–5) to identify zero-rating anomalies and structural class behaviors.
  * Encoded categorical features safely for machine learning ingestion.

---

## Exploratory Data Analysis (EDA) & Key Insights

### 1. Travel Class Disparity
* **Business Class:** Demonstrates a **70.9% satisfaction rate**, benefiting from superior cabin amenities and priority boarding.
* **Economy & Eco Plus:** Suffer from high churn risk, showing **only 39.4% and 42.7% satisfaction rates**, respectively.

### 2. Service Quality Impact Gaps
Comparing mean rating differences between satisfied and dissatisfied customer cohorts highlights where operational improvements yield maximum return:

| Service Feature | Dissatisfied Mean | Satisfied Mean | Rating Gap ($\Delta$) | Business Priority |
| :--- | :---: | :---: | :---: | :--- |
| **Inflight entertainment** | 2.61 | 4.02 | **+1.41** | Critical Driver |
| **Ease of Online booking** | 2.85 | 3.99 | **+1.13** | High Priority |
| **Online support** | 2.96 | 3.98 | **+1.02** | High Priority |
| **On-board service** | 2.97 | 3.87 | **+0.90** | Medium Priority |
| **Online boarding** | 2.87 | 3.75 | **+0.88** | Medium Priority |
| **Leg room service** | 3.05 | 3.84 | **+0.79** | Medium Priority |
| **Seat comfort** | 2.47 | 3.15 | **+0.68** | Medium Priority |

> **Key Business Takeaway:** In-flight media options and seamless digital touchpoints (online booking and support) are the strongest operational differentiators between customer retention and churn.

---

## Machine Learning Pipeline & Modeling
To ensure scientific rigor and eliminate **data leakage**, all preprocessing, encoding, and imputation steps were integrated inside strict training/testing pipelines:
* **Train-Test Split:** 80/20 stratified split (`stratify=y`) to maintain class balance (`54.7%` satisfied vs. `45.3%` dissatisfied).
* **Models Trained:** Logistic Regression (Linear Baseline), Gradient Boosting Classifier, and Random Forest Classifier.

---

## Model Evaluation & Benchmarking

| Classification Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 83.7% | 0.910 | 0.84 | 0.84 | 0.84 |
| **Gradient Boosting** | 92.7% | 0.982 | 0.93 | 0.93 | 0.93 |
| **Random Forest (Champion)** | **95.9%** | **0.993** | **0.96** | **0.96** | **0.96** |

* **Champion Model Selection:** The **Random Forest Classifier** was selected for production deployment because its ensemble architecture captures non-linear service rating interactions more effectively than linear baselines.

---

## Feature Importance & Interpretability
Feature ranking derived from tree-based Gini importance confirms the primary drivers of satisfaction:
1. **Inflight entertainment** (~18.9% importance)
2. **Seat comfort** (~13.8% importance)
3. **Ease of Online booking** (~8.1% importance)
4. **Online support** (~5.9% importance)
5. **Customer Type (Disloyal)** (~4.7% importance)

---

## Streamlit Interactive Web Application
An interactive multi-page web application (`app.py`) is included to showcase the findings dynamically:
* **Page 1 — Executive Dashboard:** High-level KPIs, total passenger volume, sentiment ratios, and class filters.
* **Page 2 — Service Deep Dive:** Interactive rating inspector allowing per-feature drill-downs and rating distribution visualizations.
* **Page 3 — ML Insights:** Model performance tables, metric matrices, and feature importance rankings.
* **Page 4 — Live Prediction Tool:** Interactive sliders and dropdowns allowing users to simulate custom passenger profiles and receive real-time Random Forest sentiment classifications.

---

## Repository Structure

```text
├── dataset/
│   └── Invistico_Airline.csv         # Cleaned airline passenger dataset
├── notebooks/
│   └── Invistico_Airline.ipynb       # Complete EDA, statistical tests & modeling notebook
├── app.py                            # Multi-page Streamlit application script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation