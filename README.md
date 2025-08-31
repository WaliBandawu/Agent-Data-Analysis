# Data Science Assistant with LangChain

This project is an AI-powered **Data Science Assistant** that leverages OpenAI’s GPT models and LangChain to interact with datasets, perform analysis, preprocessing, feature selection, and machine learning tasks. It provides a conversational interface for users to manage and explore their CSV datasets efficiently.

---

## Features

* **Dataset Management**

  * List all available datasets in a folder
  * Load and preview datasets
* **Data Analysis**

  * Generate comprehensive summary statistics
  * Clean datasets (handle missing values, encode categorical features)
  * Feature selection for model building
* **Machine Learning**

  * Train machine learning models (classification or regression)
  * Automatic task type detection based on the target variable

---

## Requirements

* Python 3.10+
* Libraries:

  * `langchain`
  * `langchain-openai`
  * `dotenv`
  * `pandas`
  * `numpy`
  * `scikit-learn`

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/WaliBandawu/Agent-Data-Analysis/
cd <repo_folder>
```

2. Create a `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Alternatively, the script will prompt you to enter the key interactively if it is not set.

---

### Example Queries

* `List all datasets`
* `Load dataset "sales_data.csv"`
* `Generate summary for dataset "customer_data.csv"`
* `Clean dataset "marketing.csv" with target "purchase"`
* `Select top 10 features from "sales.csv" with target "revenue"`
* `Train model on "sales.csv" with target "revenue"`

---

## How It Works

1. **Tool Wrappers:** Each dataset or ML-related function is wrapped as a `Tool` using `Tool.from_function`.
2. **Prompt Template:** Defines the AI assistant behavior and instructions for tool usage.
3. **LLM Integration:** Uses GPT-4o-mini to understand user queries and select the appropriate tool.
4. **Agent Execution:** `AgentExecutor` manages the flow, runs tools, and returns both results and intermediate reasoning steps.

---

## Project Structure

```
.
├── tools/
│   ├── dataset_tools.py       # Functions to list/load datasets
│   ├── analysis_tools.py      # Functions to summarize datasets
│   └── ml_tools.py            # Functions to clean, select features, and train models
├── data_agent.py              # Main agent initialization and executor
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables (OpenAI API key)
```

---

## Notes

* The agent automatically determines if a target variable corresponds to a classification or regression task.
* Always provide JSON strings when passing multiple parameters to tools like `train_model`, `clean_data`, and `feature_selection`.
