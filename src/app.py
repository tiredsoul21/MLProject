import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mean Radius, Mean Texture, Mean Perimeter, Mean Area, Mean Smoothness,
# Mean Compactness, Mean Concavity and Mean Concave Points
data = pd.read_csv('./data/cancer_dataset_wpbc.csv', sep = ',')

print("----------------Data shape---------------")
print(data.shape)

print("\n----------------Data Types---------------")
print(data.dtypes)

print("\n----------------Data head---------------")
print(data.head())


#print header keys
print("\n----------------Data keys---------------")
print(data.keys())

# Filter data to only include the features we want to use
features = data[['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
                 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points']]
features1 = data[['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area']]
features2 = data[['Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points']]

# Save the histogram of each feature to a file
features1.hist(figsize=(12, 12))
plt.savefig('./data/features1.png')
features2.hist(figsize=(12, 12))
plt.savefig('./data/features2.png')

#print and save table
print("\n----------------Data describe---------------")
print(features.describe())
features.describe().to_csv('./data/features_describe.csv')

# Assess 'Outcome'
outcome = data['Outcome'].describe()
print(outcome)

# Print Correlations
corr = features.corr()
print("\n----------------Correlation---------------")
print(corr)

# Print the heatmap
# plt.figure(figsize=(12,12))
# sns.heatmap(corr, cmap='RdBu_r', annot=True, vmax=1, vmin=-1)
# plt.savefig('./data/correlation_heatmap.png')

#Compare SE Paremeter and Mean Parameter
compare = data[['Mean Perimeter', 'SE Perimeter']]
corr = compare.corr()
print("\n----------------Parameter Correlation---------------")
print(corr)

# Make a scatter plot cause .61 doesn't say much
plt.clf()
plt.scatter(data['Mean Perimeter'], data['SE Perimeter'])
plt.xlabel('Mean Perimeter')
plt.ylabel('SE Perimeter')
plt.savefig('./data/parameter_compare.png')

# Convert outcome to binary N = 0, R = 1... 
data['Outcome'] = data['Outcome'].map({'N': 0, 'R': 1})

# Groub of Mean Area by Outcome in same plot size 6x8
plt.clf()
data.groupby('Outcome')['Mean Area'].hist(alpha=0.4)
plt.xlabel('Mean Area')
plt.legend(['N', 'R'])
plt.savefig('./data/mean_area_by_outcome.png')

# Normalize Mean area and scatter plot with Outcome
plt.clf()
meanAreaNorm = (data['Mean Area'] - data['Mean Area'].mean()) / data['Mean Area'].std()
plt.scatter(meanAreaNorm, data['Outcome'])
plt.xlabel('Mean Area')
plt.ylabel('Outcome')
plt.title('Mean Area vs Outcome Decision Boundary')
# add vertical line for 0.305576*MA -0.924867 == 0 ==> MA = 3.026634945
plt.axvline(x=3.026634945, color='red')
plt.legend(['Outcome','Decision Boundary'])
plt.savefig('./data/mean_area_outcome_scatter.png')

num_cols = [        "Mean Radius",
        "Mean Perimeter",
        "Mean Area",
        "Mean Fractal Dimension",
        "SE Texture",
        "SE Perimeter",
        "SE Area",
        "Worst Radius",
        "Worst Perimeter",
        "Worst Area",
        "Tumor Size",
        "Lymph Node Status"]
# Loop through and boxplot each variable against the outcome
for col in num_cols:
    plt.clf()
    sns.boxplot(x="Outcome", y=col, data=data)
    plt.savefig('./data/boxplot_' + col + '.png')

# Groub of Lymph Node Status by Outcome in same plot size 6x8
plt.clf()
data.groupby('Outcome')['Lymph Node Status'].hist(alpha=0.4)
plt.xlabel('Lymph Node Status')
plt.legend(['N', 'R'])
plt.savefig('./data/lns_by_outcome.png')