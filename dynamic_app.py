import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

lr = joblib.load("linear_dynamic.pkl")
ra = joblib.load("forest_dynamic.pkl")
xg = joblib.load("xgb_dynamic.pkl")
scaler = joblib.load("scaler_dynamic.pkl")

le_location = joblib.load("le_location.pkl")
le_loyalty = joblib.load("le_loyalty.pkl")
le_time = joblib.load("le_time.pkl")
le_vehicle = joblib.load("le_vehicle.pkl")

def add_features(df):
    df = df.copy()
    df['riders_per_driver'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1)
    df['past_ride_ratio'] = df['Number_of_Past_Rides'] / (df['Number_of_Riders'] + 1)
    return df

def  process():
    try:
        a=int(Number_of_Riders.get())
        b=int(Number_of_Drivers.get())
        c=le_location.transform([Location_Category.get().lower()])[0]
        d=le_loyalty.transform([Customer_Loyalty_Status.get().lower()])[0]
        e=int(Number_of_Past_Rides.get())
        f=float(Average_Ratings.get())
        g=le_time.transform([Time_of_Booking.get().lower()])[0]
        h = le_vehicle.transform([Vehicle_Type.get().lower()])[0]
        i=int(Expected_Ride_Duration.get())
        
        user_input=pd.DataFrame([[a,b,c,d,e,f,g,h,i]],columns=["Number_of_Riders","Number_of_Drivers","Location_Category",
                                           "Customer_Loyalty_Status","Number_of_Past_Rides","Average_Ratings",
                                           "Time_of_Booking","Vehicle_Type","Expected_Ride_Duration"])
        
        user_input = add_features(user_input)

        user_input_scaled=scaler.transform(user_input)
        
        pred_lr = lr.predict(user_input_scaled)[0]
        pred_rf = ra.predict(user_input)[0]
        pred_xg = xg.predict(user_input)[0]


        label1.config(text=f"Prediction Results (Estimated Ride Cost):\n"
                   f"  1. Linear Regression: ₹{pred_lr:.2f}\n"
                   f"  2. Random Forest:     ₹{pred_rf:.2f}\n"
                   f"  3. XGboost:           ₹{pred_xg:.2f}")

 
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

def clear():
    Number_of_Riders.delete(0,tk.END)
    Number_of_Drivers.delete(0,tk.END)
    Location_Category.delete(0,tk.END)
    Customer_Loyalty_Status.delete(0,tk.END)
    Number_of_Past_Rides.delete(0,tk.END)
    Average_Ratings.delete(0,tk.END)
    Time_of_Booking.delete(0,tk.END)
    Vehicle_Type.delete(0,tk.END)
    Expected_Ride_Duration.delete(0,tk.END)
    
    
root=tk.Tk()
root.title("Dynamic Ride Cost Predictor 🚕")
root.geometry("500x800")
root.configure(bg="#f4f4f4")

title = tk.Label(root, text="Smart Dynamic Pricing System", font=("Arial", 16, "bold"), bg="#065a43")
title.pack(pady=10)

tk.Label(root,text="enter Number_of_Riders:", font=("Arial", 16, "bold"), bg="#e60707").pack(pady=3)
Number_of_Riders=tk.Entry(root)
Number_of_Riders.pack(pady=5)

tk.Label(root,text="enter Number_of_Drivers:", font=("Arial", 16, "bold"), bg="#F51010").pack(pady=3)
Number_of_Drivers=tk.Entry(root)
Number_of_Drivers.pack(pady=5)

tk.Label(root,text="enter Location_Category:", font=("Arial", 16, "bold"), bg="#C61818").pack(pady=3)
Location_Category=tk.StringVar(root)
Location_Category=ttk.Combobox(root,text=Location_Category,values=["Urban","Suburban","Rural"])
Location_Category.set("Urban")
Location_Category.pack(pady=5)

tk.Label(root,text="enter Customer_Loyalty_Status:", font=("Arial", 16, "bold"), bg="#e91818").pack(pady=3)
Customer_Loyalty_Status=tk.StringVar(root)
Customer_Loyalty_Status=ttk.Combobox(root,text=Customer_Loyalty_Status,values=["Silver","Regular","Gold"])
Customer_Loyalty_Status.set("Silver")
Customer_Loyalty_Status.pack(pady=5)

tk.Label(root,text="enter  Number_of_Past_Rides:", font=("Arial", 16, "bold"), bg="#e91212").pack(pady=3)
Number_of_Past_Rides=tk.Entry(root)
Number_of_Past_Rides.pack(pady=5)

tk.Label(root,text="enter Average_Ratings:", font=("Arial", 16, "bold"), bg="#e71010").pack(pady=3)
Average_Ratings=tk.Entry(root)
Average_Ratings.pack(pady=5)

tk.Label(root,text="enter Time_of_Booking:", font=("Arial", 16, "bold"), bg="#d51010").pack(pady=3)
Time_of_Booking=tk.StringVar(root)
Time_of_Booking=ttk.Combobox(root,text=Time_of_Booking,values=["Night","Evening","Afternoon","Morning"])
Time_of_Booking.set("Night")
Time_of_Booking.pack(pady=5)

tk.Label(root,text="enter le_vehicle:", font=("Arial", 16, "bold"), bg="#d11010").pack(pady=3)
Vehicle_Type=tk.StringVar(root)
Vehicle_Type=ttk.Combobox(root,text=Vehicle_Type,values=["Premium","Economy"])
Vehicle_Type.set("Premium")
Vehicle_Type.pack(pady=5)

tk.Label(root,text="enter Expected_Ride_Duration:", font=("Arial", 16, "bold"), bg="#e11010").pack(pady=3)
Expected_Ride_Duration=tk.Entry(root)
Expected_Ride_Duration.pack(pady=5)     
        
tk.Button(root,text="click",command=process,bg="green").pack(pady=5)
tk.Button(root,text="clear",command=clear,bg="red").pack()


label1 = tk.Label(root, 
                  text="Ans:", 
                  bg="#1a1917", 
                  fg="#E7E3D6", 
                  font=("Arial", 10, "bold"),
                  wraplength=400,   
                  justify=tk.LEFT,  
                  relief=tk.SUNKEN, 
                  bd=1, 
                  padx=10, 
                  pady=10)

label1.pack(pady=10, fill='x', padx=20)
 
root.mainloop()                
        
