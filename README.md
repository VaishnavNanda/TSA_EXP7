# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 


### AIM:
To Implementat an Auto Regressive Model using Python for Meta Stock Price

### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.

   
### PROGRAM
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


file_path = 'Meta_Stock_price.csv'
data = pd.read_csv(file_path)

data['time'] = pd.to_datetime(data['time'], format='%d-%m-%Y %H:%M')
data.set_index('time', inplace=True)

close_prices = data['close']

weekly_close_prices = close_prices.resample('W').mean()

result = adfuller(weekly_close_prices.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

train_size = int(len(weekly_close_prices) * 0.8)
train, test = weekly_close_prices[:train_size], weekly_close_prices[train_size:]

fig, ax = plt.subplots(2, figsize=(8, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

ar_model = AutoReg(train.dropna(), lags=13).fit()

ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

mse = mean_squared_error(test, ar_pred)
print(f'Mean Squared Error (MSE): {mse}')

plt.figure(figsize=(10, 4))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('Train, Test, and AR Model Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


```
### OUTPUT:

GIVEN DATA
![image](https://github.com/user-attachments/assets/c35095af-f392-4fd2-afca-cd202438ad78)

PACF - ACF
![image](https://github.com/user-attachments/assets/53dc58ee-97e6-425c-be94-1b20379c8ec0)
![image](https://github.com/user-attachments/assets/684d0671-27b8-4880-8387-3564918a8bc7)

PREDICTION
![image](https://github.com/user-attachments/assets/74218b34-db5a-4d56-a006-abd59c17d18a)

FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/86600162-bf4a-4bbe-baea-798081b39f15)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
