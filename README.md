\# Solar PV Digital Twin Dashboard



Machine learning system for predicting solar panel power output from environmental parameters.



\## Features

\- 🤖 RandomForest ML model with GridSearchCV tuning

\- 🌐 FastAPI REST API for predictions

\- 🎨 Glass-themed Streamlit dashboard

\- 📊 Live digital twin simulation

\- 📁 Batch CSV predictions



\## Quick Start



\### 1. Setup Environment

```bash

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

```



\### 2. Train Model

```bash

python src/train\_production.py

```



\### 3. Start API Server

```bash

uvicorn src.server:app --reload

```



\### 4. Launch Dashboard

```bash

streamlit run src/app.py

```



\## Project Structure

```

solar-pv-project/

├── data/               # Dataset and predictions

├── models/             # Trained model artifacts

├── src/                # Source code

│   ├── train\_production.py

│   ├── server.py

│   ├── app.py

│   └── ...

└── requirements.txt

```



\## Model Performance

\- Test MAE: ~45W

\- Test R²: ~0.95

\- Training samples: 60/20/20 split



\## Author

Dhruv Bansal - College Final Year Project 2025
