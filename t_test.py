import numpy as np
from scipy.stats import t

p_mean = float(input("Enter the Population_mean: "))
s_mean = float(input("Enter the Sample_mean: "))
std = float(input("Enter the Sample Standard Deviation: "))
n = int(input("Enter the Sample Size: "))
alp = float(input("Enter the Alpha value: "))

test_type = input("Enter Test Type (two-tailed/left-tailed/right-tailed): ").lower()
print("")


df = n - 1

if test_type == "two-tailed":
    t_critical = t.ppf(1 - alp/2, df)
    print(f"t Critical Value: ±{abs(t_critical):.4f}")
elif test_type == "left-tailed":
    t_critical = t.ppf(alp, df)
    print(f"t Critical Value: {t_critical:.4f} (Left-tailed)")
elif test_type == "right-tailed":
    t_critical = t.ppf(1 - alp, df)
    print(f"t Critical Value: {t_critical:.4f} (Right-tailed)")
else:
    print("Invalid test type. Choose from 'two-tailed', 'left-tailed', 'right-tailed'")
    exit()


t_cal = (s_mean - p_mean) / (std / np.sqrt(n))
print(f"\nt Calculated: {t_cal:.4f}\n")


reject = False
if test_type == "two-tailed":
    if abs(t_cal) > abs(t_critical):
        reject = True
elif test_type == "left-tailed":
    if t_cal < t_critical:
        reject = True
elif test_type == "right-tailed":
    if t_cal > t_critical:
        reject = True
        
if reject:
    print("Since t_calculated falls in critical region:")
    print("⛔ Reject the Null Hypothesis")
else:
    print("Since t_calculated does not fall in critical region:")
    print("✅ Fail to Reject the Null Hypothesis")
