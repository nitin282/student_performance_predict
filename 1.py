import pandas as pd
data=pd.read_csv("exams.csv")
print(data)
data.isnull().sum()
data["race/ethnicity"]
count_group=data["race/ethnicity"].value_counts()
print(count_group)
x_bar=count_group.index
x_bar
y_bar=count_group.values
y_bar
import matplotlib.pyplot as plt
plt.bar(x_bar,y_bar)
plt.legend("lower left")
plt.xlabel("group",color="red")
plt.ylabel("number of group",color="red")
student_score=data[["math score","reading score","writing score"]]
print(student_score)
total=student_score.sum(axis=0)
total
plt.pie(total, labels=total.index, startangle=90, autopct="%1.1f%%")
plt.title("Total Scores Distribution")
plt.show()
from sklearn.preprocessing import LabelEncoder
dataset=data.copy()
dataset_encoded=LabelEncoder()
dataset["Gender_encode"]=dataset_encoded.fit_transform(dataset["gender"])
print(dataset[["Gender_encode"]])
dataset.to_csv("encoded_dataset.csv", index=False)
print("Encoded dataset saved successfully as 'encoded_dataset.csv'")
from sklearn.preprocessing import StandardScaler,MinMaxScaler
X=dataset[["reading score","writing score"]]
y=dataset["math score"]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(dataset[["writing score","reading score"]])
print("StandardScaler is:")
print(X_scaled)

print("-------------------------------")
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(dataset[["writing score","reading score"]])
print("MinMaxScaler is:")
print(X_scaled)

from sklearn.model_selection import train_test_split
X=dataset[["reading score","writing score"]]
y=dataset["math score"]
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train shape:",X_train)
print("X_test shape:",X_test)
print("y_train shape:",y_train)
print("y_test shape:",y_test)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
X= dataset[["reading score", "writing score", "Gender_encode", "lunch_encode", "test_encode","race_encode"]]
y=dataset["math score"]
model.fit(X,y)
reading_score=int(input("Enter reading score: "))
writing_score=int(input("Enter writing score: "))
Gender_encode=int(input("enter the gender"))
lunch_encode=int(input("enter the lunch"))
test_encode=int(input("enter the test preparation course"))
rece_encode=int(input("enter the race/ethnicity"))
predicted_score=model.predict([[reading_score,writing_score,Gender_encode,lunch_encode,test_encode,rece_encode]])
print("Predicted math score:",predicted_score)
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
y_pred=model.predict(X)
mae=mean_absolute_error(y,y_pred)
mse=mean_squared_error(y,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y,y_pred)
print("R-squared (R2):",r2)
print("Mean Absolute Error (MAE):",mae)
print("Mean Squared Error (MSE):",mse)
print("Root Mean Squared Error (RMSE):",rmse)
print("Model Accuracy (approx):", round(r2 * 100, 2), "%")
import joblib
joblib.dump(model, "math_score_model.joblib")
print("Model saved successfully as 'math_score_model.joblib'")