import numpy as np
import pandas as pd
from scipy.stats import norm



p_mean=float(input("Enter the Population_mean:"))
s_mean=float(input("Enter the Sample_mean:"))
std=float(input("Enter the Stander_deviation:"))
n=int(input("Enter the Sample:"))

alp=float(input("Enter the Aplha_value:"))


test_type = input("Enter Test Type (one-tailed/two-tailed): ").lower()
print("")

if test_type == "two-tailed":
    Z_table = norm.ppf(1 - alp/2)
else:
    Z_table = norm.ppf(1 - alp)
print("Z_table:+-(",abs(Z_table),")")

Z_cal=(s_mean-p_mean)/(std/(np.sqrt(n)))
print("")
print(" Z_calculated:",Z_cal)
print("")

if( abs(Z_cal) > abs(Z_table)):
    print("So |Z_calculated| > |Z_table|:")
    print("SO, We reject the null Hypothesis:")
    print("")
    
else:
    print("So |Z_calculated| < |Z_table|:")
    print("We fail to reject the NULL_Hypothesis:")
    print("")