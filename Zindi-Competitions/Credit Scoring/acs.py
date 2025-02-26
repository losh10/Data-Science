
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# %%
base_path = os.path.join('', 'data')
train_path = os.path.join(base_path, 'Train.csv')
test_path = os.path.join(base_path, 'Test.csv')
e_indicators = os.path.join(base_path, 'economic_indicators.csv')

# %%
indicators = pd.read_csv(e_indicators)
indicators.columns = indicators.columns.str.lower().str.replace(' ', '_')
indicators.head()

# %%
indicators.isnull().sum()

# %%
# Train and test datasets
train_df = pd.read_csv(train_path)
train_df.columns = train_df.columns.str.lower().str.replace(' ', '_')
 
test_df = pd.read_csv(test_path)
test_df.columns = test_df.columns.str.lower().str.replace(' ', '_')
 
print(train_df.shape, test_df.shape)

# %%
train_df.isnull().sum()

# %% [markdown]
# - There are no missing values from the train dataset

# %%
test_df.isnull().sum()

# %% [markdown]
# - there are no missing values from the test dataset

# %%
def date_modifier(df):
    import datetime
    from datetime import datetime

    df['disbursement_date'] = pd.to_datetime(df['disbursement_date'])
    df['due_date'] = pd.to_datetime(df['due_date'])

    # Extract the day, week, month and year for disbursement date
    
    df['disbursement_week'] = df['disbursement_date'].dt.isocalendar().week
    df['disbursement_month'] = df['disbursement_date'].dt.month
    df['disbursement_year'] = df['disbursement_date'].dt.year
    df['disbursement_day'] = df['disbursement_date'].dt.day
    df.drop('disbursement_date', axis=1, inplace=True)
    return df

# %%
train_df = date_modifier(train_df)

# %% [markdown]
# ## EDA

# %%
target = 'target'
labels = ['0', '1']
churn_counts = train_df[target].value_counts()
fig, ax = plt.subplots(1,2, figsize=(14,6))

ax[0].pie(churn_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
ax[0].set_title('Pie Chart')

ax[1].bar(churn_counts.index, churn_counts.values)
ax[1].set_xticks([0,1])
ax[1].set_xticklabels(labels)
ax[1].set_ylabel('Number of customers')
ax[1].set_title('Bar Chart')

fig.suptitle('Customer Churn Distribution Overview', fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Correlation plots
plt.figure(figsize=(16,5))
train_df.corr()[target].sort_values(ascending=False).plot(kind='bar')
plt.show()

# %% [markdown]
# - There is not much correlation between the features and the target

# %% [markdown]
# In this part we are looking at the train dataset and try to clean the data before proceeding to model development

# %%
train_df.info()

# %%
# Separating the columns into numerical and categorical columns
numerical_cols = [train_df.select_dtypes(include=['float64', 'int64']).columns]
cat_cols = [train_df.select_dtypes(include=['object']).columns]
print(numerical_cols, end='\n')
print('')
print(cat_cols)

# %% [markdown]
# ### 1. Numberical variables

# %%
# Distributions of the numeric features

num_cols_graph = ['total_amount', 'total_amount_to_repay', 'duration', 'amount_funded_by_lender', 'lender_portion_funded', 'lender_portion_to_be_repaid', 'target']
def analyze_variables(dataframe, variables):
    """
    Analyze a list of variables in a DataFrame by plotting histograms and boxplots,
    and calculating basic statistics like min, max, mean, and skewness.

    """
    for variable in variables:
        print(f"Analysis for '{variable}':\n")
        
        # Histogram
        dataframe[variable].plot(kind='hist', title=f'Histogram of {variable}')
        plt.show()

        # Boxplot
        dataframe[variable].plot(kind='box', title=f'Boxplot of {variable}')
        plt.show()
        
        # Statistics
        min_val = dataframe[variable].min()
        max_val = dataframe[variable].max()
        mean_val = dataframe[variable].mean()
        skew_val = dataframe[variable].skew()
        
        print(f"Minimum: {min_val}, Maximum: {max_val}, Average: {mean_val}")
        print(f"Skewness: {skew_val}\n")
        
analyze_variables(train_df, num_cols_graph)


# %% [markdown]
# From the visualisations we can come to these conclusions:
# 1. Most of the variables are highly positively skewed
# 2. There is presence of outliers
# 3. The target variable exhibits a significant imbalance in its distribution. 
# - we will now look at the best way of dealing with these problems

# %% [markdown]
# #### 1.  Handling skewness in the data

# %% [markdown]
# ### 2. Categorical

# %%
cat_cols = ['loan_type', 'due_date','new_versus_repeat']
cat_cols

# %%
train_df['loan_type'].value_counts().plot(kind='bar')
plt.show()

# %%
aggregate_total_amount = train_df.groupby('loan_type')['total_amount'].sum().reset_index()
plt.figure(figsize=(15,6))
plt.bar(aggregate_total_amount['loan_type'], aggregate_total_amount['total_amount'])
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# `Type_1` is the most common loan type and has the highest amount of disbursed funds

# %%
train_df['new_versus_repeat'].value_counts().plot(kind='bar')
plt.show()

# %% [markdown]
# The majority of the people have a history of taking loans. Only a few number of people are new applicants

# %%
categorical = ['loan_type', 'new_versus_repeat']

# %%
for i in categorical:
    print(i)
    print(train_df[i].unique())
    print()

# %%
repeat_loan_rate = train_df[train_df.loan_type == 'Repeat Loan'].target.mean()
new_loan_rate = train_df[train_df.loan_type == 'New Loan'].target.mean()

# %%
global_default = train_df.target.mean()

# %%
for c in categorical:
    print(c)
    df_group = train_df.groupby(c).target.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_default
    df_group['risk'] = df_group['mean'] / global_default
    display(df_group.sort_values(ascending=False, by='risk'))
    print()
    print()

# %% [markdown]
# ### Binning

# %%
labels = ['0-31', '32-60', '61-92', '93-123', '124-155','156-186', '187-218','219-250', '251-281', '282-313', '314-344', '345-375', '375-730' , 'Above 730']
bins = [0, 31, 60, 92, 123, 155, 186, 218, 250, 281, 313, 344, 375, 730, 2000]

# %%
train_df['duration_bins'] = pd.cut(train_df.duration, bins, labels = labels, include_lowest=True)

# %%
train_df.head()

# %%
train_df['duration_bins'].value_counts() / len(train_df)

# %% [markdown]
# Most of the duration is wihin 1 year
# - most of them are short term loans with 98% having a duration of 31 days

# %%
sns.barplot(data=train_df, x='disbursement_month', y='total_amount')
plt.show()

# %% [markdown]
# there are relatively few loans taken from month 7 - 11
# high loans for the period from December - July
# possible reasons:
# - for aricultural purposes (farming season usually starts in december,harvesting usually in june and july)
# - festive celebrations
# - School loans

# %%
sns.barplot(data=train_df, x='disbursement_day', y='total_amount')

# %%
train_df.head()

# %% [markdown]
# ### Domain knowledge

# %%
# creating new features
def new_features(df):
    df['duration_months'] = df['duration'] / 30
    df['customer_loan_count'] = df.groupby('customer_id')['tbl_loan_id'].transform('count')
    # df['loan_type_duration'] = df['loan_type'].astype(str) + '_' + df['duration'].astype(str)
    df['customer_loyalty'] = pd.cut(df['customer_loan_count'], bins=[0,1,3,float('inf')], labels=['new', 'repeat', 'loyal'])
    # df['roi'] = (df['lender_portion_to_be_repaid'] - df['amount_funded_by_lender']) / df['amount_funded_by_lender']
    return df

train_df = new_features(train_df)
    
    

# %%
train_df.info()

# %%
# preparing for modeling
cols_drop = ['id', 'customer_id', 'lender_id', 'country_id', 'tbl_loan_id', 'due_date', 'disbursement_week', 'disbursement_year', 'disbursement_day']
train_data = train_df.drop(cols_drop, axis=1)


# %%
train_data.info()

# %%
# cols_ohe = ['loan_type', 'new_versus_repeat', 'duration_bins', 'loan_type_duration']
# train_data[]
ohe = pd.get_dummies(train_data)
ohe.info()

# %% [markdown]
# ## Modeling

# %%
from sklearn.model_selection import train_test_split

df_copy = ohe.copy()
df_copy = df_copy.sample(frac=1)

df_full_train, df_test = train_test_split(df_copy, test_size=0.2, random_state=1)
x_train, x_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = (x_train.target).values
y_val = (x_val.target).values

del x_train['target']
del x_val['target']


# %%
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=True)

# %%
X_train = dv.fit_transform(x_train.to_dict(orient='records'))

dv.get_feature_names_out()

# %%
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='liblinear', random_state=17)
log_reg.fit(X_train, y_train)




# %%
X_val = dv.transform(x_val.to_dict(orient='records'))

# %%
# Modeling functioins

def calculate_churn_metrics(classifier, classifier_desc, x_val, y_val):
    y_pred_val = classifier.predict(x_val)
    # metrics
    val_accuracy = accuracy_score(y_val,y_pred_val)
    val_precision = precision_score(y_val, y_pred_val)
    val_recall = recall_score(y_val, y_pred_val)
    val_f1 = f1_score(y_val, y_pred_val, average='binary')
    
    
    print(f'{classifier_desc}:')
    print(f'precision: {val_precision:.4f}, recall: {val_recall:.4f}, f1_score: {val_f1:.4f}')
    print(f'accuracy: {val_accuracy:.4f}')
    print('')
    print(classification_report(y_val,y_pred_val))
    cm =  confusion_matrix(y_val, y_pred_val)
    print(cm)
    return val_accuracy, val_precision, val_recall, val_f1
    
    
    
    

# %%
def calculate_show_roc_auc(classifier, classifier_desc, x_val, y_val):
    # get predicted probabilities
    y_pred_proba = classifier.predict_proba(x_val)[:, 1]
    
    # calculate roc curve for validation set
    fpr_val, tpr_val, threshold_val = roc_curve(y_val, y_pred_proba)
    
    # calculate auc for validation set
    roc_auc_val = roc_auc_score(y_val, y_pred_proba)
    
    # plot roc curve fro validation set
    plt.figure(figsize=(8,6)) 
    plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label=f'ROC curve(AUC = {roc_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')   
    plt.ylabel('true Positive Rate')
    plt.title(f'{classifier_desc}: Receiver Operating Characteristic(ROC Curve)')
    plt.legend(loc='lower right')
    plt.show()
    return roc_auc_val

# %%
calculate_churn_metrics(log_reg, 'LogisticRegression', x_val=X_val, y_val=y_val)

# %%
calculate_show_roc_auc(log_reg, 'LogisticRegression', x_val=X_val, y_val=y_val)

# %% [markdown]
# Hyperparameter tuning

# %%
scores = []
for C in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
        lr = LogisticRegression(C=C, random_state=17)
        model = lr.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)
        scores.append((C, rmse))

df_scores = pd.DataFrame(scores, columns=['C', 'rmse'])
plt.plot(df_scores.C, df_scores.rmse)
plt.show()

# %% [markdown]
# The default value of 1 is suitable for this case therefore we will continue to use it

# %%
# results of the logistic regression training 
lr_accuracy, lr_precision, lr_recall, lr_f1 = calculate_churn_metrics(log_reg, 'LogisticRegression', x_val=X_val, y_val=y_val)

# %%
lr_roc = calculate_show_roc_auc(log_reg, 'LogisticRegression', x_val=X_val, y_val=y_val)

# %% [markdown]
# ## RandomForestClassifier

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# %%
calculate_churn_metrics(rfc, 'RandomForestClassifier', x_val=X_val, y_val=y_val)

# %%
calculate_show_roc_auc(rfc, 'RandomForestClassifier', x_val=X_val, y_val=y_val)

# %% [markdown]
# Hyperparameter tuning

# %%
scores = []
for d in [10, 15, 20, 25]:
    for n in range(100, 201, 20):
        rf = RandomForestClassifier(n_estimators=n, random_state=17, n_jobs=1, max_depth=d)
        model = rf.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred_val)
        scores.append((d, n, rmse))
        # print(d, n, round(rmse, 4))

df_scores = pd.DataFrame(scores, columns=['max_depth', 'n_estimators', 'rmse'])
for d in [10, 15, 20, 25]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.rmse, label=f'max_depth={d}')

plt.legend()
plt.show()

# %% [markdown]
# max_depth = 25
# n_estimators = 150

# %%
rfc_tuned = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=17)
rfc_tuned.fit(X_train, y_train)

# %%
rfc_tuned_accuracy, rfc_tuned_precision, rfc_tuned_recall, rfc_tuned_f1 = calculate_churn_metrics(rfc_tuned, 'RandomForestClassifier Tuned', x_val=X_val, y_val=y_val)

# %%
rfc_tuned_roc = calculate_show_roc_auc(rfc_tuned, 'RandomForestClassifier tuned',x_val=X_val,y_val=y_val)

# %%


# %%
from xgboost import XGBClassifier

# %%
xgb_clf = XGBClassifier(n_estimators=100, random_state=17)
xgb_clf.fit(X_train, y_train)

# %%
calculate_churn_metrics(xgb_clf, 'XGBClassifier', x_val=X_val, y_val=y_val)

# %%
calculate_show_roc_auc(xgb_clf, 'XGBClassifier', x_val=X_val, y_val=y_val)

# %%
from sklearn.model_selection import GridSearchCV

# %%
# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3,6,9,12],
    'learning_rate': [0.01, 0.1, 0.2] 
}
grid_search = GridSearchCV(
    estimator = xgb_clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3
)
grid_search.fit(X_train, y_train)
print(f'best parameters{grid_search.best_params_}')
print(f'best accuracy: {grid_search.best_score_}')

# %%
xgb_best = XGBClassifier(max_depth=6, n_estimators=200, learning_rate=0.2, random_state=17)
xgb_best.fit(X_train, y_train)

# %%
xgb_accuracy, xgb_precision, xgb_recall, xgb_f1 = calculate_churn_metrics(xgb_best, 'XGBClassifier Tuned', x_val=X_val, y_val=y_val)

# %%
xgb_roc = calculate_show_roc_auc(xgb_best, 'XGBClassifier Tuned', x_val=X_val, y_val=y_val)

# %%
# Data for the bar chart
models = ['Logistic Regression', 'Random Forest', 'XGBClassifier']
final_accuracy = [lr_accuracy, rfc_tuned_accuracy, xgb_accuracy]
final_precision = [lr_precision, rfc_tuned_precision, xgb_precision]
final_recall = [lr_recall, rfc_tuned_recall, xgb_recall]
final_f1_score = [lr_f1, rfc_tuned_f1, xgb_f1]
final_roc_auc = [lr_roc, rfc_tuned_roc, xgb_roc]

bar_width = 0.15

# Set the positions of the bars on the x-axis
r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(r1, final_accuracy, bar_width, label='Accuracy')
bars2 = ax.bar(r2, final_precision, bar_width, label='Precision')
bars3 = ax.bar(r3, final_recall, bar_width, label='Recall')
bars4 = ax.bar(r4, final_f1_score, bar_width, label='F1-score')
bars5 = ax.bar(r5, final_roc_auc, bar_width, label='ROC AUC')

# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison')
ax.set_xticks([r + bar_width for r in range(len(models))])
ax.set_xticklabels(models)
ax.legend(loc='lower right')

# Add value labels on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)
autolabel(bars4)
autolabel(bars5)

# Display the chart
plt.tight_layout()
plt.show()


