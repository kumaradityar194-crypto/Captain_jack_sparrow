import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

datasheet=pd.read_csv(r"c:\Users\kumar\OneDrive\Desktop\Daatasheet\dynamic_pricing.csv")
print(datasheet.head(10))
print("\n")
print(datasheet.shape)

missing=datasheet.isnull().sum()
print(missing)

print(datasheet.info())
print(datasheet.describe())

def add_features(df):
    df = df.copy()
    df['riders_per_driver'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1)
    df['past_ride_ratio'] = df['Number_of_Past_Rides'] / (df['Number_of_Riders'] + 1)
    return df

datasheet = add_features(datasheet)

categorical_cols = ["Location_Category","Customer_Loyalty_Status","Time_of_Booking","Vehicle_Type"]
for col in categorical_cols:
    datasheet[col] = datasheet[col].str.lower()

from sklearn.preprocessing import LabelEncoder

le_location = LabelEncoder().fit(datasheet["Location_Category"])
le_loyalty = LabelEncoder().fit(datasheet["Customer_Loyalty_Status"])
le_time = LabelEncoder().fit(datasheet["Time_of_Booking"])
le_vehicle = LabelEncoder().fit(datasheet["Vehicle_Type"])

datasheet["Location_Category"] = le_location.transform(datasheet["Location_Category"])
datasheet["Customer_Loyalty_Status"] = le_loyalty.transform(datasheet["Customer_Loyalty_Status"])
datasheet["Time_of_Booking"] = le_time.transform(datasheet["Time_of_Booking"])
datasheet["Vehicle_Type"] = le_vehicle.transform(datasheet["Vehicle_Type"])


print(datasheet.head(10))
print(datasheet.info())

x = datasheet.drop("Historical_Cost_of_Ride", axis=1)
y=datasheet["Historical_Cost_of_Ride"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
ra_x=scaler.fit_transform(x_train)
ra_xt=scaler.transform(x_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print("Accuracy in liner regression..:",lr.score(x_test,y_test)*100,"%")
lr_p=lr.predict(x_test)

from sklearn.ensemble import RandomForestRegressor
ra=RandomForestRegressor(n_estimators=100, random_state=42)
ra.fit(x_train,y_train)
ra_p=ra.predict(x_test)
print("Accuracy in liner Random Forest.:",ra.score(x_test,y_test)*100,"%")

from sklearn.tree import plot_tree

from xgboost import XGBRegressor
xg=XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse')
xg.fit(x_train,y_train)
xg_p=xg.predict(x_test)
print("Accuracy in Xgboost..:",xg.score(x_test,y_test)*100,"%")

a=int(input("Enter the Number_of_Riders:"))
b=int(input("Enter the Number_of_Drivers "))
c=input("Enter the Location_Category:").strip().lower()
d=input("Enter the Customer_Loyalty_Status:").strip().lower()
e=int(input("Enter the  Number_of_Past_Rides:"))
f=float(input("Enter the  Average_Ratings:"))
g=input("Enter the Time_of_Booking:").strip().lower()
h=input("Enter the Vehicle_Type:").strip().lower()
i=int(input("Enter the Expected_Ride_Duration:"))

c1 = le_location.transform([c])[0]
d1 = le_loyalty.transform([d])[0]
g1 = le_time.transform([g])[0]
h1 = le_vehicle.transform([h])[0]


user_input=pd.DataFrame([[a,b,c1,d1,e,f,g1,h1,i]],columns=["Number_of_Riders","Number_of_Drivers","Location_Category","Customer_Loyalty_Status","Number_of_Past_Rides","Average_Ratings","Time_of_Booking","Vehicle_Type","Expected_Ride_Duration"])

user_input = add_features(user_input)

user_input_scaled = scaler.transform(user_input[x_train.columns])

pred=lr.predict(user_input_scaled)
pred1=ra.predict(user_input)
pred2=xg.predict(user_input)
print("\n")

print("--- Predicted Historical Cost of Ride ---")
print("\n")
print("Predicticted Historical_Cost_of_Ride in Lr:",pred)
print("\n")
print("Predicted Historical_Cost_of_Ride in Random forest:",pred1)
print("\n")
print("Predicted Historical_Cost_of_Ride in Xgboost:",pred2)
print("\n")

print("Mean_square_error lr:",mean_squared_error(y_test,lr_p)*100,"%")
print("R2_square error in in lr",r2_score(y_test,lr_p)*100,"%")

print("Mean_square_error Forest:",mean_squared_error(y_test,ra_p)*100,"%")
print("R2_square error in in forest",r2_score(y_test,ra_p)*100,"%")

print("Mean_square_error Xgboost:",mean_squared_error(y_test,xg_p)*100,"%")
print("R2_square error in in xgboost",r2_score(y_test,xg_p)*100,"%")


lp = r2_score(y_test, lr_p)
rp = r2_score(y_test, ra_p)
xp = r2_score(y_test, xg_p)


models = ['Linear Regression', 'Random Forest', 'XGBoost']
r2_scores = [lp, rp, xp]

plt.figure(figsize=(8,5))
sns.barplot(x=models, y=r2_scores, palette='viridis')
plt.title("Model Performance Comparison (R² Score)", fontsize=14)
plt.ylabel("R² Score")
plt.xlabel("Models")
plt.ylim(0, 1)
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(y_test, color='blue', label='Actual Values', kde=True)
sns.histplot(xg_p, color='red', label='Predicted (XGBoost)', kde=True)
plt.title("Actual vs Predicted Ride Cost Distribution")
plt.legend()
plt.show()

import joblib



joblib.dump(lr, "linear_dynamic.pkl")
joblib.dump(ra, "forest_dynamic.pkl")
joblib.dump(xg, "xgb_dynamic.pkl")
joblib.dump(scaler, "scaler_dynamic.pkl")

joblib.dump(le_location, "le_location.pkl")
joblib.dump(le_loyalty, "le_loyalty.pkl")
joblib.dump(le_time, "le_time.pkl")
joblib.dump(le_vehicle, "le_vehicle.pkl")


