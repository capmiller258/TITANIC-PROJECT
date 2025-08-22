import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "train (1).csv"
df = pd.read_csv(file_path)

# Display the first few rows and column information
print("First 5 rows of the DataFrame:")
print(df.head())
print("\nInformation about the DataFrame:")
print(df.info())


# Select the required columns
df_clean = df[['PassengerId', 'Survived', 'Sex', 'Age']].copy()

# Fill missing 'Age' values with the mean
df_clean['Age'].fillna(df_clean['Age'].mean(), inplace=True)

# Create a bar chart for survival by sex
sex_survival = df_clean.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
ax = sex_survival.plot(kind='bar', figsize=(8, 6), rot=0)
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Number of People')
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('survival_by_sex.png')
plt.show()

# Create a histogram for age distribution of survivors vs. non-survivors
plt.figure(figsize=(10, 7))
plt.hist(df_clean[df_clean['Survived'] == 1]['Age'].dropna(), bins=30, alpha=0.5, label='Survived', color='green')
plt.hist(df_clean[df_clean['Survived'] == 0]['Age'].dropna(), bins=30, alpha=0.5, label='Did Not Survive', color='red')
plt.title('Age Distribution of Survivors vs. Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.legend(title='Survival Status')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.show()

# Save the cleaned dataframe to a CSV file
#df_clean.to_csv('cleaned_titanic_data.csv', index=False)
#print("Cleaned data saved to cleaned_titanic_data.csv")





