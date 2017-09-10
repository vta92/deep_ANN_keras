#uses python3
#data processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')
#indepdendent and dependent variables
x = data.iloc[:,3:13].values #slicing from credit column to the the next to last
y = data.iloc[:,13].values #1d outcome of 0/1 (binary results).  'Gender' columns to categorical (independent variables)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_country = LabelEncoder()
labelencoder_x_gender = LabelEncoder()

x[:,1] = labelencoder_x_country.fit_transform(x[:,1]) #France, Germ, Spain = (0,1,2)
x[:,2] = labelencoder_x_gender.fit_transform(x[:,2]) #female,male = (0,1) 

onehotencoder = OneHotEncoder(categorical_features=[1]) #index 1, country
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:] #country after onehotencoder

from sklearn.model_selection import train_test_split
#splitting array/matrices into random test/train subsets. 20% data set tested.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Features scaling to help ease computation. Transform data set to which mean = 0, stdev =1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

###############################################################################
#ANN implementation
import keras
from keras.models import Sequential #initialize ann
from keras.layers import Dense #build layers
from keras.layers import Dropout #regularization

#initializing ANN
classifier = Sequential() #defined as a sequence of layers

#adding first layer, and first hidden layer. 6 hidden nodes (11 in and 1 out. we're taking avg), with dropout
classifier.add(Dense(units=6,init="glorot_uniform", activation="relu", input_shape=(11,)))
classifier.add(Dropout(rate=0.1))
#second hidden layer for deep learning
classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation="relu"))
classifier.add(Dropout(rate=0.1))

#output layer
classifier.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid"))

#compiling ann.
#adam = stoichastic grad; use binary_crossentropy for 2 output sigmoid loss. category_crossentropy for multiple with sigmoid
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting from training set. Batch_size for the S.G. updating weight, epochs
classifier.fit(x_train,y_train,batch_size=10, epochs=100)

#probability of each customer having a chance to leave the bank!
y_prediction = classifier.predict(x_test)

y_prediction = (y_prediction >0.5) #change to T/F

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_prediction)

accuracy = (cm[0,0] + cm[1,1])/2000 #accuracy of predictions on new observations
print(accuracy)

##############################################################################
#new input of France, 600, male, 40 years old, 3 years, 60k, 2 prods, has credit card, active member, 50k customer
#horizontal vector for input. Same order as the feature matrix, x.
#same teting scale as the x_test, not the training scale.
new_input = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(new_input > 0.5)


##############################################################################
#K-fold Cross Validation with parameters tunning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
from keras.optimizers import TFOptimizer
from keras.models import Sequential
from keras.layers import Dense

#to rebuild the classifier based on previous code.
#train with k-fold validation
def build_classifier(optimizer): #optimizer input for parameter tuning
    classifier = Sequential() #defined as a sequence of layers
    classifier.add(Dense(units=6,init="glorot_uniform", activation="relu", input_shape=(11,)))
    classifier.add(Dropout(rate=0.05)) #dropout regularization of 5% of the neural nets (we only have a couple though :(
    classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation="relu"))
    classifier.add(Dropout(rate=0.05))
    #output layer
    classifier.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid"))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier

#without parameter tunning
#classifier = KerasClassifier(build_fn = build_classifier,batch_size=10, nb_epoch=100) #batch size = 10, epochs=100
#with parameter tuning
optimizer= {"optimizer":["Adam", "Nadam"],"batch_size":[50,100,150,200], "epochs":[50,75,100]}
classifier = KerasClassifier(build_fn= build_classifier)


accuracies = cross_val_score(estimator= classifier, X=x_train, y=y_train,cv=10, n_jobs=-1) #cv is how many folds. n_jobs = the #of cpus
grid_search = GridSearchCV(estimator=classifier, param_grid=optimizer, scoring="accuracy", cv=10) #parameter tunning with cross validation=10

grid_search = grid_search.fit(X=x_train, y=y_train)
optimal_params= grid_search.best_params_
optimal_estim = grid_search.best_estimator_

val_score_mean = accuracies.mean()
val_score_std = accuracies.std()
print(val_score_mean, val_score_std)

print(optimal_estim, optimal_params)











