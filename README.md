# AI-Powered Aviation Analytics — Aircraft Engine Predictive Maintenance

## The project

AI-Powered Aviation Analytics is a predictive-maintenance project focused on estimating the Remaining Useful Life (RUL) of aircraft engines using machine learning.

The project uses the NASA C-MAPSS FD001 dataset, which contains simulated aircraft-engine sensor measurements collected over operating cycles.

The main objective is to answer a simple question:

> How many operating cycles does an engine have remaining before failure?

Instead of relying only on fixed maintenance schedules, the project uses historical sensor behaviour to estimate the engine's remaining useful life and provide a foundation for data-driven maintenance decisions.

The project is being developed as a Conceptual Project. The core data pipeline, machine-learning models, evaluation and comparison have been completed. The remaining major component is the user-facing dashboard.

---

## The workflow

The project follows an end-to-end machine-learning workflow:

1. Load the NASA C-MAPSS FD001 data.
2. Clean and preprocess the sensor data.
3. Remove constant sensors.
4. Normalize the selected features.
5. Generate the RUL target.
6. Smooth sensor measurements.
7. Create engineered features for Random Forest.
8. Create 30-cycle sequences for LSTM.
9. Train and validate the models using engine-based splits.
10. Evaluate both models on the same 100 unseen test engines.
11. Compare the results and select the better-performing model.

The overall flow is:

Raw engine data  
→ Data preprocessing  
→ Feature engineering  
→ Random Forest / LSTM  
→ RUL prediction  
→ Evaluation  
→ Model comparison  
→ Final model selection  
→ Dashboard

---

## The data

The project uses three main files from NASA C-MAPSS FD001:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

Each engine is identified using the `unit` column and its operating history is represented using the `cycle` column.

The dataset contains:

- Operating settings
- Engine sensor measurements
- Engine operating cycles
- RUL values used for training and final evaluation

The training data contains complete engine trajectories, while the test trajectories are truncated before failure. The true RUL for the final observed point of each test engine is supplied separately in `RUL_FD001.txt`.

---

## The data pipeline

The preprocessing pipeline is implemented in:

`src/preprocessing.py`

The pipeline prepares the raw dataset before it is given to the machine-learning models.

### Sensor filtering

Sensor variance is calculated to identify constant or near-constant sensors.

Constant sensors are removed because they provide little information about changing engine condition.

### Normalization

The selected features are standardized using `StandardScaler`.

The scaler is fitted using the training data and then applied to the test data.

### RUL generation

For each training engine, the RUL is calculated from the difference between its final recorded cycle and the current cycle.

The target is capped at 125 cycles.

### Sensor smoothing

A rolling mean with a 5-cycle window is used to reduce short-term fluctuations in sensor measurements.

### Random Forest features

For selected sensors, the following rolling features are generated using a 10-cycle window:

- Rolling mean
- Rolling standard deviation
- Rolling slope

These features summarize recent sensor behaviour for the tabular Random Forest model.

### LSTM sequences

For the LSTM, the data is converted into overlapping sequences of:

`30 cycles × 18 features`

The final sequence dataset is:

`X_lstm = (17731, 30, 18)`

The corresponding targets are stored in:

`y_lstm = (17731,)`

---

## Random Forest

Random Forest is used as the project's tabular baseline model.

The implementation is contained in:

`src/models.py`

Random Forest receives the engineered sensor features created by the preprocessing pipeline.

The initial baseline uses 100 decision trees.

The model is then tuned using `GridSearchCV` with different values for:

- `n_estimators`
- `max_depth`
- `min_samples_split`

The best configuration obtained was:

- `n_estimators = 200`
- `max_depth = 20`
- `min_samples_split = 2`

The trained model is saved using Joblib.

Final Random Forest test performance on the 100 unseen test engines:

`MAE = 13.63 cycles`

`RMSE = 18.82 cycles`

`PHM08 = 662.92`

Random Forest remains an important baseline for comparison.

---

## LSTM

LSTM stands for Long Short-Term Memory and is used as the project's sequence-aware model.

Unlike Random Forest, which receives engineered tabular features, the LSTM receives a sequence of consecutive engine cycles directly.

The LSTM input is:

`30 cycles × 18 features`

The architecture is:

Input `(30, 18)`  
→ `LSTM(64)`  
→ `Dense(1)`  
→ Predicted RUL

The model is implemented using TensorFlow/Keras.

### Training

The LSTM is trained using:

- Adam optimizer
- Mean Squared Error (MSE) loss
- Batch size of 64
- Maximum of 20 epochs
- EarlyStopping based on validation loss

The train/validation split is performed by engine rather than by individual sequences.

This is important because consecutive LSTM windows from the same engine overlap heavily. Splitting randomly could allow highly similar sequences from the same engine to appear in both training and validation.

The resulting sequence sets are:

`Training = (14241, 30, 18)`

`Validation = (3490, 30, 18)`

---

## PHM08 evaluation

A PHM08-style asymmetric loss was also investigated during development.

The purpose was to test whether a specialized RUL objective would improve model performance.

The experiment performed worse than the MSE-based LSTM in terms of MAE and RMSE.

As a result:

`MSE` is used as the final training loss.

`PHM08` is retained as an additional evaluation metric.

---

## Final test evaluation

The final evaluation uses the 100 unseen test engines from C-MAPSS FD001.

For each test engine, the final 30 observed cycles are extracted and passed to the LSTM.

The predictions are then compared against the actual RUL values provided in `RUL_FD001.txt`.

The final test sequence shape is:

`(100, 30, 18)`

---

## Results

Both models were evaluated on the same 100 unseen test engines.

| Model | MAE | RMSE | PHM08 |
|---|---:|---:|---:|
| Random Forest | 13.63 | 18.82 | 662.92 |
| LSTM | **10.63** | **15.24** | **355.81** |

Lower values indicate better performance.

Based on the final test results, the LSTM achieved lower MAE, RMSE and PHM08 score than the Random Forest.

The project therefore currently uses:

**Random Forest → baseline**

**LSTM → primary RUL prediction model**

The Random Forest is retained because it provides a meaningful tabular baseline against which the sequence-based LSTM can be compared.

---

## An important lesson from evaluation

One of the most useful lessons from the project came from a result that initially looked too good.

An earlier validation experiment produced an MAE of approximately 1.36 cycles.

Instead of accepting the result immediately, the evaluation setup was examined.

The result came from evaluating terminal windows of complete validation engines where the actual RUL was often zero. This did not represent the predictive-maintenance scenario we wanted to simulate.

The evaluation approach was therefore changed, and the final comparison was performed on the 100 truncated C-MAPSS test engines.

The main lesson was:

> A good-looking metric is not necessarily a good experiment.

---

## The visualizations

The project includes three main visualizations:

### RUL distribution

A histogram is used to understand how the RUL target is distributed in the training data.

### LSTM training and validation loss

A line graph is used to understand how the model learns over epochs and identify signs of overfitting.

### Actual vs predicted RUL

A scatter plot is used to compare the LSTM's predictions against the actual RUL values for the 100 unseen test engines.

The visualization code is contained in:

`src/visualization.py`

---

## The project structure

```text
AI-Powered--Aviation--Analysis-
│
├── data/
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── visualization.py
│   ├── rf_test.py
│   └── lstm_test.py
│
├── models/
│
├── reports/
│   └── figures/
│
├── requirements.txt
└── README.md
