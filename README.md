# Data Analysis & ML Agent 🚀

This project is a **data analysis and machine learning agent** that automates feature selection and model training.  
It leverages **Zoofs** (a feature selection library based on nature-inspired optimization) and **AutoML frameworks** (e.g., LightGBM, AutoGluon) to build efficient predictive models.

---

## ✨ Features

- Automated **feature selection** using Zoofs Genetic Algorithm.
- Flexible **objective function** (log loss in current implementation).
- Supports **scikit-learn compatible models** (e.g., LightGBM, XGBoost, RandomForest).
- Integration-ready with **AutoML frameworks** (e.g., AutoGluon).
- Built for **data analysis + ML experimentation** workflows.

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/WaliBandawu/Agent-Data-Analysis.git
cd data-ml-agent
pip install -r requirements.txt
Required packages include:

zoofs

lightgbm

scikit-learn

autogluon (optional for extended AutoML support)

⚡ Usage
1. Define an Objective Function
The agent requires an objective function that trains a model and returns a score.
By default, log_loss is used:

python
Copy
Edit
from sklearn.metrics import log_loss

def objective_function_topass(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)  
    predictions = model.predict_proba(X_valid)
    return log_loss(y_valid, predictions)
2. Run Zoofs for Feature Selection
python
Copy
Edit
from zoofs import GeneticOptimization
import lightgbm as lgb

# Define model
lgb_model = lgb.LGBMClassifier()

# Initialize Zoofs optimizer
algo_object = GeneticOptimization(
    objective_function_topass,
    n_iteration=20,
    population_size=20,
    selective_pressure=2,
    elitism=2,
    mutation_rate=0.05,
    minimize=True
)

# Fit optimizer
algo_object.fit(lgb_model, X_train, y_train, X_valid, y_valid, verbose=True)

# Plot optimization history
algo_object.plot_history()

# Extract best feature subset
best_features = algo_object.best_feature_list
print("Best Features:", best_features)
📊 Workflow
Load your dataset.

Define training/validation splits.

Choose an ML model (LightGBM by default).

Optimize feature subset with Zoofs.

Evaluate results and use the best features for downstream tasks.

🔮 Roadmap
 Add support for more AutoML backends (e.g., AutoGluon).

 Build agent-like CLI for end-to-end automation.

 Extend feature selection to regression tasks.

 Add visualization dashboard.

🤝 Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

📜 License
This project is licensed under the MIT License.

