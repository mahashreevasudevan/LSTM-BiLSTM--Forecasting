# Atmospheric Visibility Forecasting Using LSTM and BiLSTM

This project focuses on building a deep learning model to forecast atmospheric visibility using both **Univariate LSTM** and **Multivariate Bidirectional LSTM (BiLSTM)** architectures. The goal is to forecast short-term atmospheric visibility based on historical data using time-series modeling techniques.


## Key Objectives:

- Forecast short-term atmospheric visibility using historical time series data.
- Compare the performance of univariate and multivariate sequence models.
- Remove noisy and nonlinear time series data using recurrent neural networks.
- Build a scalable and reusable deep learning pipeline for time-series forecasting.
- Apply best practices in data preprocessing, model training, evaluation, and visualization.


## Methodology:

### Univariate LSTM Model

Uses **historical atmospheric visibility values only** for forecasting.

- **Data Normalization:** Scaled visibility values using `MinMaxScaler`.
- **Sequence Generation:** Transformed into supervised format using sliding window.
- **Model Architecture:** LSTM → Dropout → Dense.
- **Evaluation Metrics:** MAE, RMSE.
- **Visualization:** Plots of forecasted vs. actual and training loss.

### Bidirectional LSTM (BiLSTM) Model

Uses **atmospheric visibility** and other atmospheric parameters for forecasting.

- **Data Engineering:** Multi-feature normalization and target reshaping.
- **Sequence Preparation:** Custom function to create sequences with look-back and forecasting horizon.
- **Model Architecture:** BiLSTM → BiLSTM → Dense → Dropout → Dense (output).
- **Evaluation:** Early stopping, checkpointing, and advanced metrics (MAE, MSE, RMSE, MAPE, R²).
- **Forecasting Horizon:** Multi-step forecasting support.
- **Visualization:** Actual vs. forecasted plots for validation set.

## Model Pipeline:

#### 1. Data Loading
- Load historical visibility data from a CSV file.
- For the BiLSTM model, include atmospheric paramets as well.

#### 2. Data Preprocessing
- Normalize features using `MinMaxScaler`.
- Drop rows used for future prediction (for validation/inference set).
- Data split into training and validation/test sets.

#### 3. Sequence Generation
- Create input sequences using a custom sliding window function.
- Define:
  - `look_back` (history window size)
  - `horizon` (forecasting window size)
- Reshape inputs into 3D arrays for LSTM model.

#### 4. Model Architecture
- **Univariate LSTM:**
  - LSTM → Dropout → Dense
- **BiLSTM:**
  - Bidirectional LSTM (2 layers) → Dense → Dropout → Dense (output)

#### 5. Model Training
- Use early stopping to prevent overfitting.
- Save best model checkpoints using validation loss.
- Train using TensorFlow’s `tf.data` pipeline for batching and shuffling.

#### 6. Prediction & Inverse Scaling
- Predict visibility values on the validation/test set.
- Invert scaled predictions to original values using fitted scalers.

#### 7. Evaluation
- Evaluate model performance using:
  - MSE, MAE, RMSE, MAPE, and R² score.
- Compare actual vs predicted results visually.

#### 8. Visualization
- Plot training/validation loss curves.
- Plot actual vs. predicted values for qualitative analysis.

## Challenges Addressed:

- Using time series data with temporal dependencies.
- Normalizing the time series to avoid scaling problems.
- Reducing overfitting with dropout and early stopping.
- Reshaping data for LSTM model input format.

## Results:

### Univariate LSTM

The univariate LSTM model captured short-term patterns in the visibility series but its forecasts on new or unseen data were less accurate. This might be due to the absence of visibility influencing parameters.

### Bidirectional LSTM (BiLSTM)

The BiLSTM model significantly improved forecast quality by learning from both historical visibility and related parameters. This was better at modeling temporal dependencies and gave accurate forecasts without overfitting. 

## Technology and Tools:

- **Languages & Libraries**:
  - Python
  - NumPy, Pandas
  - Matplotlib, Seaborn
  - Scikit-learn
  - Keras, TensorFlow
  - Statsmodels, SciPy
    
## Note:
The dataset used in this project is confidential. Hence, dataset, model hyperparameters, plots, and the error metrics are not shown.

