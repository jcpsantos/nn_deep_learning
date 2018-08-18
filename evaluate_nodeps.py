import warnings
import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#df_nn = pd.read_csv('train_100k.csv')
#df_truth = pd.read_csv('train_100k.truth.csv')

#df_nn.drop(['id'], axis=1, inplace=True)
#df_truth.drop(['id'], axis=1, inplace=True)

#df_nn.to_csv('new_train_100k.csv')
#df_truth.to_csv('new_train_100k.truth.csv')

df_nn = np.loadtxt('train_100k.csv', delimiter=",", skiprows=1)
df_truth = np.loadtxt('train_100k.truth.csv', delimiter=",", skiprows=1)

x = df_nn[:, 1:22]
y = df_truth[:, 1:3]

#create model 
model = Sequential()
model.add(Dense(24, input_dim=20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

#compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(x, y, epochs = 150, batch_size=1000)

#evaluate the model
slope_intercept = model.evaluate(x, y)

print("\n%s: %.2f%%" % (model.metrics_names[1], slope_intercept[1]*100))
#print("Slope mse: %s" % np.mean(slope_ses))
#print("Intercept mae: %s" % np.mean(intercept_aes))

pred = np.loadtxt('test_100k.csv', delimiter=",", skiprows=1)    

xPredicted = pred[:, 1:22]

# calculate predictions
predictions = model.predict(xPredicted)

rounded = [round(xPredicted[0]) for xPredicted in predictions]
print(rounded)


