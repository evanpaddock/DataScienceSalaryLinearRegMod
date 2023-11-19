import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("Salary_Data.csv")

df.dropna(inplace=True, axis=0)

df["gender_male"] = df["Gender"].apply(lambda x: 1 if x == "Male" else 0)
df["gender_female"] = df["Gender"].apply(lambda x: 1 if x == "Female" else 0)

df["education_bach"] = df["Education Level"].apply(
    lambda x: 1 if x == "Bachelor's" else 0
)
df["education_mast"] = df["Education Level"].apply(
    lambda x: 1 if x == "Master's" else 0
)
df["education_phd"] = df["Education Level"].apply(lambda x: 1 if x == "PhD" else 0)

salary_corr = df.corr(numeric_only=True)["Salary"].sort_values().abs()[:-1]
print(salary_corr)

ols_model = ols("Salary ~ Q('Years of Experience') + Age + education_phd + education_mast + education_bach + gender_male + gender_female", df).fit()
df["salary_pred"] = ols_model.predict(df[["Salary", "Years of Experience", "Age", "education_phd", "education_mast", "education_bach", "gender_male", "gender_female"]])

print(ols_model.summary())

mse = mean_squared_error(df["Salary"], df["salary_pred"])
mae = mean_absolute_error(df["Salary"], df["salary_pred"])

print(mse, mae)

plt.figure()
plt.scatter(df["Years of Experience"], df["Salary"], s=1, color="Black", alpha=0.8)
plt.scatter(df["Years of Experience"], df["salary_pred"], s=1, color="Red", alpha=0.8)
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
