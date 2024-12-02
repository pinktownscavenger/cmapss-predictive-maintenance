import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('CMAPSSDATA/train_FD001.txt', header=None, delimiter=' ')
df = df.iloc[:,:-2]
df.columns = ['Unit', 'Cycles', 'OpSet1', 'OpSet2', 'OpSet3'] + [f'Sensor{i}' for i in range(1, 22)]

# Calculating End of Life (EOL) and Remaining Useful Life (RUL)
EOL = [df[df['Unit'] == unit]['Cycles'].values[-1] for unit in df['Unit']]
df['EOL'] = EOL
df['RUL'] = df['EOL'] - df['Cycles']
df = df.drop(columns=['EOL', 'Unit'])

df.nunique(axis=0)

# Boxplots for sensor readings
plt.figure(figsize=(15, 21))
for i in np.arange(1, 25):
    temp = df.iloc[:, i]
    plt.subplot(5, 6, i)
    plt.boxplot(temp)
    plt.title(df.columns[i])
plt.show()

# Correlation heatmap
corrmat = df.corr()
plt.figure(figsize=(58, 58))
sns.set(font_scale=4, font="Times New Roman")
g = sns.heatmap(df[corrmat.index].corr(), cmap="RdYlGn", linewidths=0.1, annot=True, annot_kws={"size": 35})
g.set_xticklabels(g.get_xmajorticklabels(), fontsize=35)
g.set_yticklabels(g.get_xmajorticklabels(), fontsize=35)

# Removing features with low correlation to RUL
del_cols = [col for col in df.columns if abs(df[col].corr(df['RUL'])) <= 0.5]
new_df = df.drop(columns=del_cols)

# Correlation heatmap for selected features
corrmat = new_df.corr()
plt.figure(figsize=(58, 58))
sns.set(font_scale=4, font="Times New Roman")
g = sns.heatmap(new_df[corrmat.index].corr(), cmap="RdYlGn", linewidths=0.1, annot=True, annot_kws={"size": 35})
g.set_xticklabels(g.get_xmajorticklabels(), fontsize=35)
g.set_yticklabels(g.get_xmajorticklabels(), fontsize=35)

from sklearn.ensemble import ExtraTreesRegressor

# Feature importance using ExtraTreesRegressor
X = new_df.iloc[:, 0:13]
y = new_df.iloc[:, 13]
model = ExtraTreesRegressor()
model.fit(X, y)

# Plot feature importances
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)
feat_importances.nlargest(13).plot(kind='barh', fontsize=13)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

X = new_df.iloc[:, 0:-1]
y = new_df.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("RMSE on Test Set: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

y_pred_train = regressor.predict(X_train)
print("RMSE on Training Set: ", np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))

feat_imp_df = new_df[['Cycles', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor7', 'Sensor8', 'Sensor11', 'Sensor12', 
                      'Sensor13', 'Sensor15', 'Sensor17', 'Sensor20', 'Sensor21']]

number_of_features = []
test_rmse = []
train_rmse = []

for i in range(1, 13):
    number_of_features.append(i)
    X = feat_imp_df.iloc[:, 0:i]
    y = new_df.iloc[:, 13]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    test_rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    y_pred_train = regressor.predict(X_train)
    train_rmse.append(np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
