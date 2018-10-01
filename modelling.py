# # Customer defaulting likelyhood

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


# ## Data preprocessing


# Import data
data_0716 = pd.ExcelFile('data/FE data - 0716S.xlsx')


# In[203]:


all_data = pd.read_excel(data_0716, '0716SDL')
all_data.head()


# In[204]:


all_data.info()


# ## Model 1: Including item types

# In[205]:


# Drop unecessary features
features_to_drop = [
    'Mth(s)', 
    'Defaulted', 
    'Value ($)', 
    'Interest Payable ($)'
]
model_1_df = all_data.drop(features_to_drop, axis=1)


# In[206]:


# Get dummies and save dependent variable
status_list = model_1_df.Status.tolist()
model_1_df = model_1_df.drop(['Status'], axis=1)
model_1_dummied_df = pd.get_dummies(model_1_df)
model_1_dummied_df['Status'] = status_list

# In[208]:


# Get dependent and independent variable arrays
x = model_1_dummied_df.iloc[:, :-1].values
y = model_1_dummied_df.iloc[:, -1].values

# Get training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# In[209]:


# Create and fit classifier
classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)


# In[210]:


# Predict and print result
y_pred_probabilties = classifier.predict_proba(x_test)
y_pred_class = classifier.predict(x_test)
print(pd.DataFrame({
    'C%_pred': y_pred_probabilties[:, 0],
    'D%_pred': y_pred_probabilties[:, 1],
    'L%_pred': y_pred_probabilties[:, 2],
    'most_likely_status': y_pred_class,
    'actual': y_test
}))


# In[211]:


# Print accuracy score of absolute prediction
cm = confusion_matrix(y_test, y_pred_class)
print('Accuracy =', float(cm[0][0] + cm[1][1])/np.sum(cm) * 100, '%')


# In[226]:


# Pickle and save model
joblib.dump(classifier, 'models/model_1.pkl')


# ## Model 2: Excluding item types (except watch)

# In[212]:


# Drop unecessary features
features_to_drop = [
    'Mth(s)', 
    'Defaulted', 
    'Value ($)', 
    'Interest Payable ($)',
    'Anklet',
    'Bangle',
    'Bracelet',
    'Chain',
    'Earring',
    'Earstud',
    'Necklace',
    'Pendant',
    'Ring',
    'O'
]
model_2_df = all_data.drop(features_to_drop, axis=1)


# In[213]:


# Get dummies and save dependent variable
status_list = all_data.Status.tolist()
model_2_df = model_2_df.drop(['Status'], axis=1)
model_2_dummied_df = pd.get_dummies(model_2_df)
model_2_dummied_df['Status'] = status_list


# In[214]:


# Get dependent and independent variable arrays
x = model_2_dummied_df.iloc[:, :-1].values
y = model_2_dummied_df.iloc[:, -1].values

# Get training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# In[215]:


# Create and fit classifier
classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)


# In[216]:


# Predict and print result
y_pred_probabilties = classifier.predict_proba(x_test)
y_pred_class = classifier.predict(x_test)
print(pd.DataFrame({
    'C%_pred': y_pred_probabilties[:, 0],
    'D%_pred': y_pred_probabilties[:, 1],
    'L%_pred': y_pred_probabilties[:, 2],
    'most_likely_status': y_pred_class,
    'actual': y_test
}))


# In[217]:


# Print accuracy score of absolute prediction
cm = confusion_matrix(y_test, y_pred_class)
print('Accuracy =', float(cm[0][0] + cm[1][1])/np.sum(cm) * 100, '%')


# **Observations**: Model 2 actually performs worse, suggesting that the item being pawned is important

# In[227]:


# Pickle and save model
joblib.dump(classifier, 'models/model_2.pkl')


# ## Model 3: Excluding all item types

# In[218]:


# Drop unecessary features
features_to_drop = [
    'Mth(s)', 
    'Defaulted', 
    'Value ($)', 
    'Interest Payable ($)',
    'Mth(s)', 
    'Defaulted', 
    'Value ($)', 
    'Interest Payable ($)',
    'Anklet',
    'Bangle',
    'Bracelet',
    'Chain',
    'Earring',
    'Earstud',
    'Necklace',
    'Pendant',
    'Ring',
    'O',
    'W'
]
model_3_df = all_data.drop(features_to_drop, axis=1)


# In[219]:


# Get dummies and save dependent variable
status_list = all_data.Status.tolist()
model_3_df = model_3_df.drop(['Status'], axis=1)
model_3_dummied_df = pd.get_dummies(model_3_df)
model_3_dummied_df['Status'] = status_list


# In[220]:


# Get dependent and independent variable arrays
x = model_3_dummied_df.iloc[:, :-1].values
y = model_3_dummied_df.iloc[:, -1].values

# Get training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# In[221]:


# Create and fit classifier
classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)


# In[222]:


# Predict and print result
y_pred_probabilties = classifier.predict_proba(x_test)
y_pred_class = classifier.predict(x_test)
print(pd.DataFrame({
    'C%_pred': y_pred_probabilties[:, 0],
    'D%_pred': y_pred_probabilties[:, 1],
    'L%_pred': y_pred_probabilties[:, 2],
    'most_likely_status': y_pred_class,
    'actual': y_test
}))


# In[223]:


# Print accuracy score of absolute prediction
cm = confusion_matrix(y_test, y_pred_class)
print('Accuracy =', float(cm[0][0] + cm[1][1])/np.sum(cm) * 100, '%')


# **Observations**: as expected, model 3 performs as bad as model 2

# In[228]:


# Pickle and save model
joblib.dump(classifier, 'models/model_3.pkl')


# ## Model 4: Including item information, value and interest

# In[229]:


# Drop unecessary features
features_to_drop = [
    'Mth(s)', 
    'Defaulted'
]
model_4_df = all_data.drop(features_to_drop, axis=1)


# In[230]:


# Get dummies and save dependent variable
status_list = model_4_df.Status.tolist()
model_4_df = model_4_df.drop(['Status'], axis=1)
model_4_dummied_df = pd.get_dummies(model_4_df)
model_4_dummied_df['Status'] = status_list


# In[231]:


# Get dependent and independent variable arrays
x = model_4_dummied_df.iloc[:, :-1].values
y = model_4_dummied_df.iloc[:, -1].values

# Get training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# In[232]:


# Create and fit classifier
classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)


# In[233]:


# Predict and print result
y_pred_probabilties = classifier.predict_proba(x_test)
y_pred_class = classifier.predict(x_test)
print(pd.DataFrame({
    'C%_pred': y_pred_probabilties[:, 0],
    'D%_pred': y_pred_probabilties[:, 1],
    'L%_pred': y_pred_probabilties[:, 2],
    'most_likely_status': y_pred_class,
    'actual': y_test
}))


# In[234]:


# Print accuracy score of absolute prediction
cm = confusion_matrix(y_test, y_pred_class)
print('Accuracy =', float(cm[0][0] + cm[1][1])/np.sum(cm) * 100, '%')


# In[235]:


# Pickle and save model
joblib.dump(classifier, 'models/model_4.pkl')

