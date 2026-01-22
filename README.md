
# Deep Learning Predictive Maintenance for Oil & Gas Equipment
## (NASA CMAPSS – LSTM – TensorFlow)

### Project Overview
This project implements an **industrial-grade predictive maintenance system** using **Deep Learning (LSTM)** to estimate the **Remaining Useful Life (RUL)** of critical Oil & Gas equipment such as gas turbines, compressors, and rotating machinery.

The solution is aligned with **Sonatrach** and **Saudi Aramco** digital transformation, asset integrity, and reliability engineering use cases.

---
### Business Value
- Reduce unplanned shutdowns
- Optimize maintenance scheduling
- Improve asset reliability
- Support condition-based maintenance (CBM)

---
### Dataset
NASA CMAPSS Turbofan Engine Degradation Dataset  
Mapped to Oil & Gas rotating equipment behavior.

---
### Tech Stack
- Python
- TensorFlow / Keras
- LSTM (Time Series)
- Pandas / NumPy
- Scikit-learn
- Matplotlib

---
### Project Structure
```
├── data/
│   └── train_FD001.txt
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Training.ipynb
│   └── 03_Evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── sequence_generator.py
│   ├── lstm_model.py
│   └── train.py
├── requirements.txt
└── README.md
```
---
### Author
Data Scientist / Machine Learning Engineer – Oil & Gas AI
