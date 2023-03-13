# data processing
import pandas as pd
import numpy as np

# data visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning library
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets
df_brazilian_houses = pd.read_csv("datasets/brazilian_houses.csv")
df_books = pd.read_csv("datasets/goodreads_books.csv")
df_boston_houses = pd.read_csv("datasets/boston_house_prices.csv")
df_iris = pd.read_csv("datasets/iris.csv")

# Datasets information
d = {"brazilian_houses": [len(df_brazilian_houses), len(df_brazilian_houses.columns)],
     "boston_houses": [len(df_boston_houses), len(df_boston_houses.columns)],
     "iris": [len(df_iris), len(df_iris.columns)],
"books": [len(df_books), len(df_books.columns)],
}
print ("{:<18} {:<22} {:<33}".format("dataset", "number of instances", "number of attributes"))
for k, v in d.items():
    perc, change = v
    print ("{:<25} {:<25} {:<18}".format(k, perc, change))

# Histogram distribution visualisation and boxplot distribution
# (For convenience, this visualisation has been added as a comment)

# sns.displot(df_brazilian_houses['propertyTax'], kde = True)
# # label the axis
# plt.xlabel('propertyTax')
# plt.ylabel("Count")
# plt.show()
# sns.boxplot(df_brazilian_houses['propertyTax'])
# plt.show()
#
# sns.displot(df_boston_houses['CRIM'], kde = True)
# # label the axis
# plt.xlabel('CRIM')
# plt.ylabel('Count')
# plt.show()
# sns.boxplot(df_boston_houses['CRIM'])
# plt.show()
#
# sns.displot(df_iris['PetalLengthCm'], kde = True)
# # label the axis
# plt.xlabel('PetalLengthCm')
# plt.ylabel("Count")
# plt.show()
# sns.boxplot(df_iris['PetalLengthCm'])
# plt.show()
#
# sns.displot(df_books['num_pages'], kde = True)
# # label the axis
# plt.xlabel('Number of pages')
# plt.ylabel('Count')
# plt.show()
# sns.boxplot(df_books['num_pages'])
# plt.show()


# Algorithms

# Z- score algorithm
def z_score(data):
    mean = np.mean(data)
    std = np.std(data)
    threshold = 3
    outlier = []
    for i in data:
        z = (i - mean) / std
        if z > threshold or z < -threshold:
            if i not in outlier:
                outlier.append(i)
    return outlier

# Modified z- score with our optimization
def modified_z_score(df_data_copy, col, data, threshold, r):
    outlier = []
    prev_mad = 0
    for i in range(r):
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if prev_mad == mad:
            # print(i, "iterations")
            break
        for d in data:
            modified_z_scores = (0.6745 * (d - median)) / (1.4826 * mad)
            if modified_z_scores > threshold or modified_z_scores < -threshold:
                # Find the row indices where the specified column equals the specified value
                rows_to_drop = df_data_copy.index[df_data_copy[col] == d].tolist()
                # Drop the rows with the specified indices
                df_data_copy = df_data_copy.drop(rows_to_drop)
                if d not in outlier:
                    outlier.append(d)
                data.remove(d)
        prev_mad = mad
    df_data_copy.to_csv('fileWithoutOutliers.csv')
    return outlier, df_data_copy

# Main function
def detect_remove_outliers(dataset_name, df_dataset, column, r, nd):
    df_data_copy = pd.DataFrame(df_dataset.copy())
    specific_column = df_dataset[column]
    specific_column_list = specific_column.tolist()
    outliers, m = modified_z_score(df_data_copy, column, specific_column_list, 3.5, r)
    if nd:
        outliers2 = z_score(specific_column_list)
        print("z score outliers", len(outliers2))
    if outliers:
        print(dataset_name, ":", len(outliers), "outliers at", column, "column")
    else:
        print(dataset_name, ":", "There are no outliers at", column, "column")
    return m

# We applied our algorithm to the datasets in order to detect any
# outliers in a specific column. We ran the algorithm for 10 iterations,
# as previous testing had determined that this was the maximum number of
# iterations required for convergence to the median absolute deviation (MAD) for all columns.

# choose dataset and column for detecting outliers
detect_remove_outliers("brazilian_houses", df_brazilian_houses, "propertyTax", 10, False)
# detect_remove_outliers("brazilian_houses", df_brazilian_houses, "bathroom", 10, False)
# detect_remove_outliers("brazilian_houses", df_brazilian_houses, "rentAmount", 10, False)
# detect_remove_outliers("brazilian_houses", df_brazilian_houses, "rooms", 10, False)
# detect_remove_outliers("books", df_books, "num_pages", 10, False)
# detect_remove_outliers("iris", df_iris, "PetalLengthCm", 10, False)
# detect_remove_outliers("boston_houses", df_boston_houses, "CRIM", 10, False)

#Experimental evaluation

#Linear Regression

df_houses = pd.read_csv("datasets/brazilian_houses.csv")
X = df_houses[["propertyTax"]]
Y = df_houses[["total"]]
plt.xlabel("propertyTax")
plt.ylabel("total")
plt.title("propertyTax and total Relationship")
plt.scatter(df_houses.rentAmount, df_houses.total)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)
plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_test, y_pred_test, color='green', linewidth=2)
plt.title("Actual vs Predicted Values (Test Set)")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.show()

r2score = r2_score(Y_test, y_pred_test)
print("Training set R2 SCORE: ", r2score)

# Linear Regression with more dimensions

# 1. brazilian_houses dataset

# Perform linear regression on the dataset **before removing any outliers

df_houses = pd.read_csv("datasets/brazilian_houses.csv")

# defined X and Y columns
Y_column = 'total'
all_columns = df_houses.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = df_houses[X_columns]
Y = df_houses[Y_column]

# split to test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
print("MSE without removing outliers: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
print("RMSE without removing outliers: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
print("R2 SCORE without removing outliers: ", r2score)

# Perform linear regression on the dataset **after removing outliers by our algorithm

df_brazilian_houses = pd.read_csv("datasets/brazilian_houses.csv")

data_frame_after_remove = detect_remove_outliers("brazilian_houses", df_houses, "propertyTax", 10, False)

# defined X and Y columns
Y_column = 'total'
all_columns = data_frame_after_remove.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = data_frame_after_remove[X_columns]
Y = data_frame_after_remove[Y_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
print("MSE with outliers removed by our alg: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
print("RMSE with outliers removed by our alg: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
print("R2 SCORE with outliers removed by our alg: ", r2score)

# Perform linear regression on the dataset **after removing outliers by the original modified z score algorithm

df_brazilian_houses = pd.read_csv("datasets/brazilian_houses.csv")

data_frame_after_remove = detect_remove_outliers("brazilian_houses", df_houses, "propertyTax", 1, False)

# defined X and Y columns
Y_column = 'total'
all_columns = data_frame_after_remove.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = data_frame_after_remove[X_columns]
Y = data_frame_after_remove[Y_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
print("MSE with outliers removed by the original modified z score: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
print("RMSE with outliers removed by the original modified z score: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
print("R2 SCORE with outliers removed by the original modified z score: ", r2score)

# 2. boston_houses dataset

# Perform linear regression on the dataset **before removing any outliers

# boston houses Dataset contains 13 features that are used
# to predict the median value (MEDV) of owner-occupied homes in a given suburb

# before removing outliers
df_boston = pd.read_csv("datasets/boston_house_prices.csv")

# defined X and Y columns
Y_column = 'MEDV'
all_columns = df_boston.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = df_boston[X_columns]
Y = df_boston[Y_column]

# split to test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE without removing outliers: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE without removing outliers: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE without removing outliers: ", r2score)

# Perform linear regression on the dataset **after removing outliers by our algorithm

df_boston = pd.read_csv("datasets/boston_house_prices.csv")

data_frame_after_remove = detect_remove_outliers("boston_houses", df_boston_houses, "CRIM", 10, False)

# defined X and Y columns
Y_column = 'MEDV'
all_columns = data_frame_after_remove.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = data_frame_after_remove[X_columns]
Y = data_frame_after_remove[Y_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE with outliers removed by our alg: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE with outliers removed by our alg: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE with outliers removed by our alg: ", r2score)

# Perform linear regression on the dataset **after removing outliers by the original modified z score algorithm

df_boston = pd.read_csv("datasets/boston_house_prices.csv")

data_frame_after_remove = detect_remove_outliers("boston_houses", df_boston_houses, "CRIM", 1, False)

# defined X and Y columns
Y_column = 'MEDV'
all_columns = data_frame_after_remove.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = data_frame_after_remove[X_columns]
Y = data_frame_after_remove[Y_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE with outliers removed by the original modified z score: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE with outliers removed by the original modified z score: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE with outliers removed by the original modified z score: ", r2score)

# 3. iris dataset

# Perform linear regression on the dataset **before removing any outliers

# The original Iris Dataset contains 3 species at 'Species' column: Iris-setosa,Iris-versicolor,
# Iris-virginica. Since the data we can work with is numeric, we change it into 0, 1, 2 correspondly.

# before removing outliers
df_iris = pd.read_csv("datasets/iris.csv")

# defined X and Y columns
Y_column = 'Species'
all_columns = df_iris.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = df_iris[X_columns]
Y = df_iris[Y_column]

# split to test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE without removing outliers: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE without removing outliers: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE without removing outliers: ", r2score)

# Perform linear regression on the dataset **after removing outliers by our algorithm

df_iris = pd.read_csv("datasets/iris.csv")

data_frame_after_remove = detect_remove_outliers("iris", df_iris, "PetalLengthCm", 10, False)

# defined X and Y columns
Y_column = 'Species'
all_columns = data_frame_after_remove.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = data_frame_after_remove[X_columns]
Y = data_frame_after_remove[Y_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE with outliers removed by our alg: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE with outliers removed by our alg: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE with outliers removed by our alg: ", r2score)

# Perform linear regression on the dataset **after removing outliers by the original modified z score algorithm

df_iris = pd.read_csv("datasets/iris.csv")

data_frame_after_remove = detect_remove_outliers("iris", df_iris, "PetalLengthCm", 1, False)

# defined X and Y columns
Y_column = 'Species'
all_columns = data_frame_after_remove.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = data_frame_after_remove[X_columns]
Y = data_frame_after_remove[Y_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE with outliers removed by the original modified z score: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE with outliers removed by the original modified z score: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE with outliers removed by the original modified z score: ", r2score)

# 4. books dataset

# Perform linear regression on the dataset **before removing any outliers

df_books = pd.read_csv("datasets/goodreads_books.csv")

# defined X and Y columns
Y_column = 'average_rating'
all_columns = df_books.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = df_books[X_columns]
Y = df_books[Y_column]

# split to test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE without removing outliers: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE without removing outliers: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE without removing outliers: ", r2score)

# Perform linear regression on the dataset **after removing outliers by our algorithm

df_books = pd.read_csv("datasets/goodreads_books.csv")

data_frame_after_remove = detect_remove_outliers("books", df_books, "num_pages", 10, False)

# defined X and Y columns
Y_column = 'average_rating'
all_columns = data_frame_after_remove.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = data_frame_after_remove[X_columns]
Y = data_frame_after_remove[Y_column]

# split to test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE with outliers removed by our alg: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE with outliers removed by our alg: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE with outliers removed by our alg: ", r2score)

# Perform linear regression on the dataset **after removing outliers by the original modified z score algorithm

df_books = pd.read_csv("datasets/goodreads_books.csv")

data_frame_after_remove = detect_remove_outliers("books", df_books, "num_pages", 1, False)

# defined X and Y columns
Y_column = 'average_rating'
all_columns = data_frame_after_remove.columns.to_list()
X_columns = list(set(all_columns) - set([Y_column]))

# create X and Y
X = data_frame_after_remove[X_columns]
Y = data_frame_after_remove[Y_column]

# split to test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test ,y_pred_test, squared=True)
# print("MSE with outliers removed by the original modified z score: ", mse)

rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
# print("RMSE with outliers removed by the original modified z score: ", rmse)

r2score = r2_score(Y_test, y_pred_test)
# print("R2 SCORE with outliers removed by the original modified z score: ", r2score)

# Boxplot distribution before and after removing outliers by our algorithm
# (For convenience, this visualisation has been added as a comment)

# df_after_remove = detect_remove_outliers("brazilian_houses", df_brazilian_houses, "propertyTax", 10, False)
# f, axes = plt.subplots(1, 2)
# sns.boxplot(data=df_brazilian_houses["propertyTax"],ax=axes[0])
# sns.boxplot(data=df_after_remove["propertyTax"],ax=axes[1])
# axes[0].set_title('Before removing outliers')
# axes[1].set_title('After removing outliers')
# f.tight_layout()
# plt.show()
#
# df_after_remove = detect_remove_outliers("boston_houses", df_boston_houses, "CRIM", 10, False)
# f, axes = plt.subplots(1, 2)
# sns.boxplot(data=df_boston_houses["CRIM"],ax=axes[0])
# sns.boxplot(data=df_after_remove["CRIM"],ax=axes[1])
# axes[0].set_title('Before removing outliers')
# axes[1].set_title('After removing outliers')
# f.tight_layout()
# plt.show()
#
#
# df_after_remove = detect_remove_outliers("books", df_books, "num_pages", 10, False)
# f, axes = plt.subplots(1, 2)
# sns.boxplot(data=df_books["num_pages"],ax=axes[0])
# sns.boxplot(data=df_after_remove["num_pages"],ax=axes[1])
# axes[0].set_title('Before removing outliers')
# axes[1].set_title('After removing outliers')
# f.tight_layout()
# plt.show()