# Titanic Survival Predictor

A machine learning classification project that predicts passenger survival aboard the RMS Titanic based on demographic and voyage-related attributes.

## Project Overview

The objective of this project is to build binary classification models to predict whether a passenger survived (`1`) or perished (`0`) during the Titanic disaster.

The project covers an end-to-end data science and machine learning workflow:

- Data cleaning and handling missing values (`Age`, `Cabin`, `Embarked`, `Fare`)
- Exploratory Data Analysis (EDA) and univariate/bivariate visualizations
- Feature engineering: title extraction from names (`Title`), family size calculation (`FamilySize`), and one-hot encoding
- Feature selection and matrix alignment across train and test sets
- Model training across multiple algorithms
- Cross-validation and hyperparameter optimization using `GridSearchCV`
- Feature importance evaluation
- Final test set prediction and Kaggle-format submission generation (`submission.csv`)

---

## Dataset

The dataset originates from the classic Kaggle Titanic competition:

- **Training Records:** 891 passengers
- **Test Records:** 418 passengers
- **Target Variable:** `Survived` (0 = Did not survive, 1 = Survived)

### Features

| Feature | Description | Type / Values |
|---|---|---|
| `PassengerId` | Unique identifier for each passenger | Numerical |
| `Pclass` | Ticket class (socio-economic status proxy) | 1st, 2nd, 3rd |
| `Name` | Full name of the passenger | Text (used to extract `Title`) |
| `Sex` | Gender of the passenger | `male`, `female` |
| `Age` | Age in years (imputed by title/class group medians) | Numerical |
| `SibSp` | Number of siblings / spouses aboard | Numerical |
| `Parch` | Number of parents / children aboard | Numerical |
| `Ticket` | Ticket number string | Text |
| `Fare` | Passenger fare | Numerical |
| `Cabin` | Cabin number | Alphanumeric |
| `Embarked` | Port of embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

### Engineered Features

- **`Title`**: Extracted titles (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) from the passenger's name.
- **`FamilySize`**: Total family members on board (`SibSp + Parch + 1`).

---

## Models & Evaluation

Multiple classification models were trained and benchmarked on a validation split:

### 1. Logistic Regression (Baseline)
- **Accuracy:** 81.01%
- **Precision:** 0.786
- **Recall:** 0.743
- **F1 Score:** 0.764

### 2. Decision Tree Classifier
- Explored tree depths from 1 through 10 to balance bias and variance.
- Best test accuracy achieved around depth 3:
  - **Accuracy:** 79.89%
  - **F1 Score:** 0.739

### 3. Random Forest Classifier (Baseline)
- Evaluated with 50 estimators:
  - **Accuracy:** 83.24%
  - **Precision:** 0.797
  - **Recall:** 0.797
  - **F1 Score:** 0.797

---

## Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used to search the parameter space for both Random Forest and Gradient Boosting models.

### Tuned Random Forest
- **Best Hyperparameters:** `n_estimators`: 200, `max_depth`: 6
- **5-Fold CV Accuracy:** 83.42%
- **Validation Set Accuracy:** 82.68%
- **Validation F1 Score:** 0.780

### Tuned Gradient Boosting Classifier
- **Best Hyperparameters:** `n_estimators`: 200, `learning_rate`: 0.05, `max_depth`: 4
- **5-Fold CV Accuracy:** 83.00%
- **Validation Set Accuracy:** 82.68%
- **Validation F1 Score:** 0.780

---

## Feature Importance

Feature importance analysis from the ensemble model showed the top drivers for passenger survival:

1. **`Fare`**: Higher-paying passengers had significantly better access to lifeboats.
2. **`Age`**: Children received evacuation priority.
3. **`Sex_male`**: Female survival rate was vastly higher due to "women and children first" protocols.
4. **`Title_Mr`**: Adult males had the lowest survival rate.
5. **`Pclass`**: 1st class passengers had higher survival rates than 2nd and 3rd class passengers.

---

## Project Structure

```text
titanic_survival_predictor/
│
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── titanic.ipynb
├── .gitignore
├── requirements.txt
├── submission.csv
└── README.md
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/utkarrshgit/titanic_survival_predictor.git
   cd titanic_survival_predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/titanic.ipynb
   ```

---

## Technologies Used

- **Language:** Python
- **Data Manipulation:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook
