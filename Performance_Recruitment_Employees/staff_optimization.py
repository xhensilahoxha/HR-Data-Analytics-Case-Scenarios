# importing necessary libraries
import pandas as pd
from pulp import *

# load dataset and save into dataframe
data = 'Performance_Recruitment_Employees/fau_medical_staff.csv'
df = pd.read_csv(data)

# fill NaN values with 0 & map 'X' to 1
df = df.fillna(0).map(lambda x: 1 if x == 'X' else x)

# select the row where 'Time Windows' column equals "Wage rate per 8h shift (EUR)"
shift_pay = df[df['Time Windows'] == "Wage rate per 8h shift (EUR)"].iloc[0, 1:4].values.astype(int)

# remove last row
df = df.iloc[:-1]
# number of time windows
time_windows = len(df.index)

# Shift matrix 3x3
shifts = df[['Shift 1', 'Shift 2', 'Shift 3']].values.astype(int)
shift_num = shifts.shape[1]

# avg. patient numbers per time slot
avg_patient_num = df['Avg_Patient_Number'].values.astype(float)

service_level = 4

# Decision variable: number of assistants per shift
num_assistants_indx = [f'shift_{j}' for j in range(shift_num)]
num_assistants = LpVariable.dicts("num_assistants", num_assistants_indx, lowBound=0, cat="Integer")

# Optimization problem
prob = LpProblem("Optimize_Medical_Staff_Allocation", LpMinimize)

# Objective: Minimize the total payment of nurses across shifts
prob += lpSum([shift_pay[j] * num_assistants[f'shift_{j}'] for j in range(shift_num)])

# Constraints: Ensure patient demand is met for each time window
for t in range(time_windows):
    prob += lpSum([shifts[t, j] * num_assistants[f'shift_{j}'] for j in range(shift_num)]) >= avg_patient_num[t] / service_level

# Solve the problem
prob.solve()

# Output
print("Problem status:", LpStatus[prob.status])
for j in range(shift_num):
    print(f"Number of assistants per shift is {j + 1}: {int(num_assistants[f'shift_{j}'].value())}")

