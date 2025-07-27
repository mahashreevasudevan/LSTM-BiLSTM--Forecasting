# Atmospheric Visibility Forecasting Using Univariate LSTM

This project focuses on building a deep learning model using Long Short-Term Memory (LSTM) networks to forecast atmospheric visibility based on historical data.

## Key Objectives:

- Forecast short-term atmospheric visibility based on historic values.
- Remove noisy and nonlinear time series data using recurrent neural networks.
- Build a scalable and adaptable pipeline for atmospheric parameters forecasting .

## Methodology:

- Data Preprocessing: Load and normalize visibility data using MinMaxScaler.
- Sequence Generation: Convert time series into supervised learning format using a sliding window (`look_back`).
- Model Training: Use an LSTM model with dropout regularization.
- Evaluation: Measure model accuracy using MAE and RMSE.
- Visualization: Plot predictions vs actuals and training/test loss curves.

## Model Pipeline:

1. **Data Loading**  
   Load raw visibility data from a CSV file.

2. **Normalization**  
   Normalize the values to a 0–1 scale to stabilize training.

3. **Train-Test Split**  
   Split the dataset into training and testing subsets.

4. **Sequence Creation**  
   Define a `look_back` window to shape data for time-series forecast.

5. **Model Architecture**  
   - LSTM layer  
   - Dropout layer for regularization  
   - Dense output layer

6. **Training**  
   Fit the model with early stopping on validation loss.

7. **Prediction & Evaluation**  
   - Forecast visibility on both train and test sets  
   - Invert normalization  
   - Evaluate performance using MAE and RMSE

8. **Visualization**  
   - Plot training/validation loss  
   - Plot actual vs forecasted values 

## Challenges Addressed:

- Using time series data with temporal dependencies.
- Normalizing the time series to avoid scaling problems.
- Reducing overfitting with dropout and early stopping.
- Reshaping data for LSTM model input format.

## Results:

- The model shows steady and low error rates on both training and testing datasets.
- Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) indicate strong generalization and accurate forecasts.
- The performance on the test set is comparable to the training set, suggesting the model is not overfitting and is capable of handling unseen data effectively.

## Technology and Tools:

- **Languages & Libraries**:
  - Python
  - NumPy, Pandas
  - Matplotlib, Seaborn
  - Scikit-learn
  - Keras (TensorFlow backend)
  - Statsmodels, SciPy
    
## Note:
The dataset used in this model is confidential. Hence, dataset, model hyperparameters, plots, and the error metrics are not shown.

