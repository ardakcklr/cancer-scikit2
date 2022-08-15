import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import metrics
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings('ignore')


# Reading 'clincial.tsv' file
cancer = pd.read_csv('clinical.tsv', sep='\t')

# Replace '--' values with NaN 
cancer = cancer.replace("'--", np.nan)
# Drop columns with only NaN values
cancer = cancer.dropna(axis=1, how='all')

# Columns which will not be used are dropped
cancer.drop(cancer.columns.difference(['case_submitter_id', 'age_at_index', 'days_to_death', 'gender', 'race',
                                       'vital_status', 'year_of_birth', 'year_of_death', 'age_at_diagnosis',
                                       'ajcc_pathologic_stage', 'icd_10_code', 'primary_diagnosis', 'prior_malignancy',
                                       'prior_treatment', 'site_of_resection_or_biopsy', 'synchronous_malignancy',
                                       'tissue_or_organ_of_origin', 'year_of_diagnosis', 'treatment_type']), 1, inplace=True)

# Rows which has a NaN value are dropped
cancer = cancer.dropna()
# Rows which has 'Not Reported' or 'not reported' values are dropped
cancer = cancer.drop([col for col in cancer.columns if cancer[col].eq("not reported").any() or
                      cancer[col].eq("Not Reported").any()], axis=1)
# Recurring rows with the same case_submitter_id are dropped except for the first iteration
cancer = cancer.drop_duplicates(subset=["case_submitter_id"], keep= 'first')
# case_submitter_id column is set as the index of the dataFrame
cancer = cancer.set_index("case_submitter_id")

# print(cancer.treatment_type)

# Determining features that are thought be necessary
# constructing a new dataframe named 'death_df' for normalization and one hot encoding operations
# vital_status column is eliminated since it only has one unique value
death_df = cancer[['age_at_index', 'days_to_death', 'age_at_diagnosis', 'icd_10_code',
                    'primary_diagnosis', 'prior_malignancy', 'prior_treatment',
                    'tissue_or_organ_of_origin', 'treatment_type']]



# Type of categorigal columns is changed to category to implement one hot encoding 
death_df['prior_treatment'] = death_df['prior_treatment'].astype('category')
death_df['prior_malignancy'] = death_df['prior_malignancy'].astype('category')
death_df['primary_diagnosis'] = death_df['primary_diagnosis'].astype('category')
death_df['icd_10_code'] = death_df['icd_10_code'].astype('category')
death_df['tissue_or_organ_of_origin'] = death_df['tissue_or_organ_of_origin'].astype('category')


# Assigning proper numerical values to data in columns, and storing them in matching columns
death_df['treatment_numeric'] = death_df['prior_treatment'].cat.codes
death_df['malignancy_numeric'] = death_df['prior_malignancy'].cat.codes
death_df['diagnosis_numeric'] = death_df['primary_diagnosis'].cat.codes
death_df['code_numeric'] = death_df['icd_10_code'].cat.codes
death_df['tissue_organ_numeric'] = death_df['tissue_or_organ_of_origin'].cat.codes


# Assigning a one hot encoder
enc = OneHotEncoder()

# Getting data from one hot encoded columns
enc_data = pd.DataFrame(enc.fit_transform(death_df[['treatment_numeric', 'malignancy_numeric', 'diagnosis_numeric',
                                                    'code_numeric', 'tissue_organ_numeric']]).toarray())




# Generating prediction dataFrame and joining death dataFrame in here
prediction_df = death_df.join(enc_data)

# Dropping NaN valued rows and columns
prediction_df = prediction_df.dropna(axis=1, how='all')
prediction_df.dropna()
# print(prediction_df)

# Normalization with MinMaxScaler on non-categorical columns
scaler = MinMaxScaler()
scaler.fit(prediction_df[['age_at_index', 'age_at_diagnosis']])
scaled = scaler.fit_transform(prediction_df[['age_at_index','age_at_diagnosis']])
scaled_df = pd.DataFrame(scaled)

prediction_df[['age_at_index']] = MinMaxScaler().fit_transform(np.array(prediction_df[['age_at_index']]).reshape(-1,1))
prediction_df[['age_at_diagnosis']] = MinMaxScaler().fit_transform(np.array(prediction_df[['age_at_diagnosis']]).reshape(-1,1))
prediction_df[['days_to_death']] = MinMaxScaler().fit_transform(np.array(prediction_df[['days_to_death']]).reshape(-1,1))

# Dropping columns with object values since they are not going to be used
prediction_df = prediction_df.drop(['icd_10_code', 'primary_diagnosis', 'prior_malignancy', 'prior_treatment',
                   'tissue_or_organ_of_origin'], axis=1)

# print(prediction_df)
# print(list(prediction_df.columns.values))

# Locating treatment_type data which will be predicted, from the x-axis to y-axis
X = prediction_df.drop('treatment_type', axis=1)
y = prediction_df.loc[:, 'treatment_type']



'''
# ///// Making prediction with Gradient Boosted Classifier model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

from sklearn.ensemble import GradientBoostingClassifier
gbt1_model = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_depth=3, random_state=1)
gbt1_model = gbt1_model.fit(X_train, y_train)
predictions = gbt1_model.predict(X_test)

model_accuracy_score = accuracy_score(y_test, predictions)
model_accuracy_score = model_accuracy_score * 100

print('Accuracy for Decision Tree model: %',"{:.2f}".format(model_accuracy_score))

model_importances = pd.DataFrame(gbt1_model.feature_importances_, index = X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)

print(model_importances)

print('Cross-validation scores for Gradient Boosting classification model: ',cross_val_score(dt1_model, X, y, cv=5), '\n')
accuracies = cross_val_score(estimator = dt1_model, X=X_train, y=y_train, cv=10)
print('Cross validation accuracy with the Gradient Boosting classification model: %', "{:.2f}".format(accuracies.mean()*100))
'''

'''
# ///// Making predictions with Decision Tree Classifier model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y, train_size=0.20)
dt1_model = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=40, random_state=1)
dt1_model = dt1_model.fit(X_train, y_train)
predictions = dt1_model.predict(X_test)

model_accuracy_score = accuracy_score(y_test, predictions)
model_accuracy_score = model_accuracy_score * 100

print('Accuracy for Decision Tree model: %',"{:.2f}".format(model_accuracy_score))

# Feature importances on model's learning
model_importances = pd.DataFrame(dt1_model.feature_importances_, index = X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)

print(model_importances)

print('Cross-validation scores for Decision Tree classification model: ',cross_val_score(dt1_model, X, y, cv=5), '\n')
accuracies = cross_val_score(estimator = dt1_model, X=X_train, y=y_train, cv=10)
print('Cross validation accuracy with the Decision Tree classification model: %', "{:.2f}".format(accuracies.mean()*100))
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

# ///// Hyperparameter optimization with GridSearchCV module for Random Forest Classifier
rf1_grid = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rf1_grid, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

print('Hyperparameter Optimization: ',CV_rfc.best_params_)

# ///// Making predictions with Random Forest Classifier model

rf1_model=RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 500, max_depth=4, criterion='gini')
rf1_model = rf1_model.fit(X_train, y_train)
predictions = rf1_model.predict(X_test)

model_accuracy_score = accuracy_score(y_test, predictions)
model_accuracy_score = model_accuracy_score * 100

print("Accuracy for Random Forest on CV data: ", "{:.2f}".format(model_accuracy_score))

model_importances = pd.DataFrame(rf1_model.feature_importances_, index = X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)

print(model_importances)

print('Cross-validation scores for Random Forest classification model: ',cross_val_score(rf1_model, X, y, cv=5), '\n')
accuracies = cross_val_score(estimator = rf1_model, X=X_train, y=y_train, cv=10)
print('Cross validation accuracy with the Random Forest classification model: %', "{:.2f}".format(accuracies.mean()*100))
