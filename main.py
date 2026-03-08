#import require python classes and packages
import pandas as pd #pandas to read and explore dataset
import numpy as np
import matplotlib.pyplot as plt #use to visualize dataset vallues
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm #SVM class
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
import os
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
#loading and displaying heart disease dataset
dataset = pd.read_csv("Dataset/predictive_maintenance.csv")
dataset
#describing dataset with details like count, mean, standard deviation of each dataset attributes
dataset.describe()
#visualizing distribution of numerical data
dataset.hist(figsize=(10, 8))
plt.title("Representation of Dataset Attributes")
plt.show()
#finding and displaying count of missing or null values
dataset.isnull().sum()
#finding & plotting graph of failure machine parts which required maintenance
#visualizing class labels count found in dataset
labels, count = np.unique(dataset['Failure_Type'].ravel(), return_counts = True)
height = count
bars = labels
y_pos = np.arange(len(bars))
plt.figure(figsize = (4, 3)) 
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Dataset Class Label Graph")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()
#visualizing product quality as number of Low, high and medium quality
#describe and plotting graph of various Product Current Quality %  found in dataset 
dataset.groupby("Type").size().plot.pie(autopct='%.0f%%', figsize=(4, 4))
plt.title("Product Type % graph")
plt.xlabel("Product Condition Type")
plt.ylabel("Condition %")
plt.show()
#visualizing tool life with different failure conditions
data = dataset.groupby(['Failure_Type', 'Type'])['Tool_wear_[min]'].sum().sort_values(ascending=False).reset_index()
sns.catplot(x="Type", y="Tool_wear_[min]", hue='Failure_Type', data=data, kind='point')
plt.title("Product Life Time Based on Failure Type and Product Condition Type")
plt.show()
dataset.groupby('Type')['Process_temperature_[K]'].plot(legend=True, figsize=(6,3))
plt.title("Process Temperature Available in All Products Quality")
plt.show()
data = dataset[['Failure_Type', 'Air_temperature_[K]', 'Rotational_speed_[rpm]', 'Torque_[Nm]']]
plt.figure(figsize=(12,4))
sns.boxplot(data=data, x='Failure_Type', y='Air_temperature_[K]', palette='rainbow')
plt.title("Machine Air Temperature Available in Normal & Failure Conditions")
plt.show()
plt.figure(figsize=(12,4))
sns.violinplot(data=data, x='Failure_Type', y='Rotational_speed_[rpm]', palette='rainbow')
plt.title("Machine Rotational speed Available in Normal & Failure Conditions")
plt.show()
#using label encoder converting non-numeric values to numeric values
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()
dataset['Product_ID'] = pd.Series(encoder1.fit_transform(dataset['Product_ID'].astype(str)))#encode all str columns to numeric
dataset['Type'] = pd.Series(encoder2.fit_transform(dataset['Type'].astype(str)))#encode all str columns to numeric
dataset['Failure_Type'] = pd.Series(encoder3.fit_transform(dataset['Failure_Type'].astype(str)))#encode all str columns to numeric
#dataset pre-processing like removing irrelevant features and selecting relevant features from the dataset
dataset.drop(['UDI', 'Target'], axis = 1,inplace=True)
print("Dataset After Cleaning & Processing")
dataset
#dataset shuffling and normalization
rul = dataset['Tool_wear_[min]'].ravel()#represents life of machine (rul = remaining useful life)
Y = dataset['Failure_Type'].ravel()#represents machine failure or normal
data = dataset.values
X = data[:,0:dataset.shape[1]-1]
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffling dataset values
X = X[indices]
Y = Y[indices]
rul = rul[indices]
#normalizing dataset values
scaler = MinMaxScaler(feature_range = (0, 1))
X = scaler.fit_transform(X)#normalize train features
print("Normalize Training Features")
print(X)
#split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print("Total records found in dataset = "+str(X.shape[0]))
print("Total features found in dataset= "+str(X.shape[1]))
print("80% dataset for training : "+str(X_train.shape[0]))
print("20% dataset for testing  : "+str(X_test.shape[0]))
#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []
#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+" Accuracy  : "+str(a))
    print(algorithm+" Precision : "+str(p))
    print(algorithm+" Recall    : "+str(r))
    print(algorithm+" FSCORE    : "+str(f))
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 3))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_tpr, ns_fpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.show()
    #training and evaluating performance of SVM algorithm
svm_cls = svm.SVC(C=50)
svm_cls.fit(X_train, y_train)#train algorithm using training features and target value
predict = svm_cls.predict(X_test) #perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("SVM Algorithm", y_test, predict)
#training and evaluating performance of decision tree algorithm
dt_cls = DecisionTreeClassifier()
dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
predict =dt_cls.predict(X_test)#perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("Decision Tree Algorithm", y_test, predict)
#training and evaluating performance of RandomForestClassifier algorithm
regressor = RandomForestRegressor()
regressor.fit(X, rul)
rf_cls = RandomForestClassifier(max_depth=10)
rf_cls.fit(X_train, y_train)#train algorithm using training features and target value
predict = rf_cls.predict(X_test)#perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("Random Forest", y_test, predict)
#training and evaluating performance of RandomForestClassifier algorithm
knn_cls = KNeighborsClassifier(n_neighbors=2)
knn_cls.fit(X_train, y_train)#train algorithm using training features and target value
predict = knn_cls.predict(X_test)#perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("KNN", y_test, predict)
#training CNN deep learning algorithm to predict factory maintenaance
#converting dataset shape for CNN comptaible format as 4 dimension array
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)
#creating deep learning cnn model object
cnn_model = Sequential()
#defining CNN layer wwith 32 neurons of size 1 X 1 to filter dataset features 32 times
cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
#defining maxpool layet to collect relevant filtered features from previous CNN layer
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
#creating another CNN layer with 16 neurons to optimzed features 16 times
cnn_model.add(Convolution2D(16, (1, 1), activation = 'relu'))
#max layet to collect relevant features
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
#convert multidimension features to single flatten size
cnn_model.add(Flatten())
#define output prediction layer
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
#compile, train and load CNN model
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train1, y_train1, batch_size = 4, epochs = 50, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")
#perform prediction on test data   
predict = cnn_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
#call this function to calculate accuracy and other metrics
calculateMetrics("CNN", y_test1, predict)
#comparison graph between all algorithms
df = pd.DataFrame([['SVM','Accuracy',accuracy[0]],['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','FSCORE',fscore[0]],
                   ['Decision Tree','Accuracy',accuracy[1]],['Decision Tree','Precision',precision[1]],['Decision Tree','Recall',recall[1]],['Decision Tree','FSCORE',fscore[1]],
                   ['Random Forest','Accuracy',accuracy[2]],['Random Forest','Precision',precision[2]],['Random Forest','Recall',recall[2]],['Random Forest','FSCORE',fscore[2]],
                   ['KNN','Accuracy',accuracy[3]],['KNN','Precision',precision[3]],['KNN','Recall',recall[3]],['KNN','FSCORE',fscore[3]],
                   ['Deep Learning CNN','Accuracy',accuracy[4]],['Deep Learning CNN','Precision',precision[4]],['Deep Learning CNN','Recall',recall[4]],['Deep Learning CNN','FSCORE',fscore[4]],
                  ],columns=['Parameters','Algorithms','Value'])
df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
plt.title("All Algorithms Performance Graph")
plt.show()
#display all algorithm performnace
algorithms = ['SVM', 'Decision Tree', 'Random Forest', 'KNN', 'Deep Learning CNN']
data = []
for i in range(len(accuracy)):
    data.append([algorithms[i], accuracy[i], precision[i], recall[i], fscore[i]])
data = pd.DataFrame(data, columns=['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE'])
data   
test_data = pd.read_csv("Dataset/testData.csv")
temp = test_data.values
#using label encoder converting non-numeric values to numeric values
test_data['Product_ID'] = pd.Series(encoder1.transform(test_data['Product_ID'].astype(str)))#encode all str columns to numeric
test_data['Type'] = pd.Series(encoder2.transform(test_data['Type'].astype(str)))#encode all str columns to numeric
#dataset pre-processing like removing irrelevant features and selecting relevant features from the dataset
test_data.drop(['UDI'], axis = 1,inplace=True)
test_data = test_data.values
test_data = scaler.transform(test_data)
#life prediction before maintenance
life = regressor.predict(test_data)
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1, 1))
#failure prediction
# ==============================
# FAILURE PREDICTION WITH ALERT
# ==============================

predict = cnn_model.predict(test_data)

print("\n=========== PREDICTIVE MAINTENANCE ALERT SYSTEM ===========\n")

for i in range(len(predict)):

    pred_class = np.argmax(predict[i])
    confidence = np.max(predict[i]) * 100
    remaining_life = 100 - (life[i] / 10)

    print("Machine Sensor Data :", temp[i])
    print("Predicted Condition :", labels[pred_class])
    print("Prediction Confidence : {:.2f}%".format(confidence))
    print("Estimated Remaining Useful Life : {:.2f}%".format(remaining_life))

    # 🚨 ALERT CONDITIONS
    if labels[pred_class] != "No Failure":
        print("🚨 ALERT: MACHINE FAILURE DETECTED!")
        print("⚠ Immediate Maintenance Required!")
    elif remaining_life < 30:
        print("⚠ WARNING: Machine life is critically low. Schedule maintenance soon.")
    else:
        print("✅ Machine operating under normal conditions.")

    print("------------------------------------------------------\n")

