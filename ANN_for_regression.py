import pandas as pd
import numpy as np


# =============================================================================
# Importing data
# =============================================================================

path = r"C:\Users\jose\Documents\Python code\DDBB\CarPricesData.pkl"
df = pd.read_pickle(path)


# =============================================================================
# Data wrangling
# =============================================================================

df.columns
df.describe()
df.dtypes


# =============================================================================
# Defining the dataset
# =============================================================================

features = ['Age', 'KM', 'Weight', 'HP', 'MetColor', 'CC', 'Doors']
target = ['Price']

x = df[features]
y = df[target]


# =============================================================================
# Standarize the data
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_fit = scaler_x.fit(x)
y_fit = scaler_y.fit(y)


mean_x = x_fit.mean_
std_x = x_fit.var_**(1/2)


x_transf = x_fit.transform(x)
y_transf = y_fit.transform(y)


# =============================================================================
# Splitting the data for training and testing
# =============================================================================

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_transf, y_transf, test_size=.3, random_state=42)


# =============================================================================
# Creating the model
# =============================================================================

from keras.layers import Dense
from keras.models import Sequential

# create ANN model

model = Sequential()

# Defining the input layer (it's the same as the first layer)
model.add(Dense(units=5, input_dim=7, kernel_initializer='normal', activation='relu'))

# Defining the second layer of the model 
model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))

# The output layer is a single fully connected node 
model.add(Dense(units=1, kernel_initializer='normal'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fitting the ANN to the training set
model.fit(x_train, y_train, batch_size=20, epochs=50, verbose=1)


# =============================================================================
# Using the model for prediction
# =============================================================================

y_pred = model.predict(x_test)




# =============================================================================
# Creating inverted transformation
# =============================================================================

x_test_inverted = scaler_x.inverse_transform(x_test)
y_test_inverted = scaler_y.inverse_transform(y_test)
y_predicted_inverted = scaler_y.inverse_transform(y_pred)

# =============================================================================
# Creating dataset
# =============================================================================

x_test_df = pd.DataFrame(x_test_inverted, columns=features)
y_test_df = pd.DataFrame(y_test_inverted, columns=target)
y_pred_df = pd.DataFrame(y_predicted_inverted, columns=['Predicted'])

df_final = pd.concat([x_test_df, y_test_df, y_pred_df], axis=1)


# =============================================================================
# Calculating MAPE
# =============================================================================


df_final["APE"] = np.abs((df_final.Price - df_final.Predicted) / df_final.Price)
mape = df_final.APE.mean()
accuracy = (1 - mape)*100
print("the accuracy of the model is" , accuracy) 


# =============================================================================
# Testing a real use
# =============================================================================

path = r"C:\Users\jose\Documents\Python code\DDBB\Test Regression ANN.xlsx"
df_use = pd.read_excel(path)


# Standarizing the data

df_st = (df_use - mean_x) / std_x

# Make the prediction as a real use

y_pred_real = model.predict(df_st)
real_pred = scaler_y.inverse_transform(y_pred_real)
print('The prediction is ', real_pred)












































