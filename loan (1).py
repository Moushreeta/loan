#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


loan_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv",index_col=False)
loan_data = loan_data.drop(['Unnamed: 0'], axis=1)
loan_data.head()


# In[3]:


test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')
test_data.head()


# In[4]:


test_data.shape


# In[5]:


loan_data['Loan_Status'].value_counts()


# In[6]:


loan_data['Loan_Status'].value_counts(normalize=True)


# In[7]:


loan_data['Loan_Status'].value_counts().plot.bar()


# In[8]:


loan_data['Gender'].value_counts(normalize=True).plot.bar(title='Gender')
plt.show()
loan_data['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.show()
loan_data['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')
plt.show()
loan_data['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
plt.show()


# In[9]:


loan_data['Dependents'].value_counts(normalize=True).plot.bar( title='Dependents')
plt.show()
loan_data['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.show()
loan_data['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()


# In[10]:


sns.distplot(loan_data['ApplicantIncome'])
plt.show()
loan_data['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# In[11]:


loan_data.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")


# In[12]:


sns.distplot(loan_data['CoapplicantIncome'])
plt.show()
loan_data['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# In[13]:


loan_data.notna()
# train.dropna()
# print(train[train['LoanAmount'].isnull()])
# train['LoanAmount'] = pd.to_numeric(train['LoanAmount'], errors='coerce')
# train = train.dropna(subset=['LoanAmount'])
# train['LoanAmount'] = train['LoanAmount'].astype(int)
sns.distplot(loan_data['LoanAmount'])
plt.show()
loan_data['LoanAmount'].plot.box(figsize=(16,5))
plt.show()


# In[14]:


Gender=pd.crosstab(loan_data['Gender'],loan_data['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()


# In[15]:


Married=pd.crosstab(loan_data['Married'],loan_data['Loan_Status'])
Dependents=pd.crosstab(loan_data['Dependents'],loan_data['Loan_Status'])
Education=pd.crosstab(loan_data['Education'],loan_data['Loan_Status'])
Self_Employed=pd.crosstab(loan_data['Self_Employed'],loan_data['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()


# In[16]:


Credit_History=pd.crosstab(loan_data['Credit_History'],loan_data['Loan_Status'])
Property_Area=pd.crosstab(loan_data['Property_Area'],loan_data['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.show()


# In[17]:


loan_data.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[18]:


bins=[0,2500,4000,6000,81000]
group=['Low','Average','High','Very high']
loan_data['Income_bin']=pd.cut(loan_data['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(loan_data['Income_bin'],loan_data['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel('ApplicantIncome')
P=plt.ylabel('Percentage')


# In[19]:


bins=[0,1000,3000,42000]
group=['Low','Average','High']
loan_data['Coapplicant_Income_bin']=pd.cut(loan_data['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(loan_data['Coapplicant_Income_bin'],loan_data['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel('CoapplicantIncome')
P=plt.ylabel('Percentage')


# In[20]:


loan_data['Total_Income']=loan_data['ApplicantIncome']+loan_data['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High','Very high']
loan_data['Total_Income_bin']=pd.cut(loan_data['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(loan_data['Total_Income_bin'],loan_data['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel('Total_Income')
P=plt.ylabel('Percentage')


# In[21]:


bins=[0,100,200,700]
group=['Low','Average','High']
loan_data['LoanAmount_bin']=pd.cut(loan_data['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(loan_data['LoanAmount_bin'],loan_data['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel('LoanAmount')
P=plt.ylabel('Percentage')


# In[22]:


# print(train.dtypes)
loan_data=loan_data.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
loan_data['Dependents'].replace('3+', 3,inplace=True)
test_data['Dependents'].replace('3+', 3,inplace=True)


# In[23]:


matrix = loan_data.corr()
f, ax = plt.subplots(figsize=(9,6))
sns.heatmap(matrix,vmax=.8,square=True,cmap="BuPu", annot = True)


# In[24]:


loan_data.isnull().sum()


# In[25]:


loan_data['Gender'].fillna(loan_data['Gender'].mode()[0], inplace=True)
loan_data['Married'].fillna(loan_data['Married'].mode()[0], inplace=True)
loan_data['Dependents'].fillna(loan_data['Dependents'].mode()[0], inplace=True)
loan_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0], inplace=True)
loan_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0], inplace=True)


# In[26]:


loan_data['Loan_Amount_Term'].value_counts()


# In[27]:


loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mode()[0], inplace=True)


# In[28]:


loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].median(), inplace=True)


# In[29]:


loan_data.isnull().sum()


# In[30]:


test_data['Gender'].fillna(loan_data['Gender'].mode()[0], inplace=True)
test_data['Married'].fillna(loan_data['Married'].mode()[0], inplace=True)
test_data['Dependents'].fillna(loan_data['Dependents'].mode()[0], inplace=True)
test_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0], inplace=True)
test_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0], inplace=True)
test_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mode()[0], inplace=True)
test_data['LoanAmount'].fillna(loan_data['LoanAmount'].median(), inplace=True)


# In[31]:



loan_data['LoanAmount_log']=np.log(loan_data['LoanAmount'])
loan_data['LoanAmount_log'].hist(bins=20)
test_data['LoanAmount_log']=np.log(test_data['LoanAmount'])


# In[32]:


loan_data=loan_data.drop('Loan_ID',axis=1)
test_data=test_data.drop('Loan_ID',axis=1)


# In[33]:


loan_data


# In[34]:



X = loan_data.drop('Loan_Status',axis=1)
y = loan_data.Loan_Status
     


# In[35]:


train = loan_data.copy()
test = test_data.copy()
     


# In[36]:


X = pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)


# In[37]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3)


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
model = LogisticRegression()
model.fit(x_train, y_train)


# In[39]:


pred_cv = model.predict(x_valid)
print('Model Accuracy = ', accuracy_score(y_valid,pred_cv))
print('Model F1-Score = ', f1_score(y_valid,pred_cv))


# In[40]:



pred_test = model.predict(test)


# In[41]:


res = pd.DataFrame(pred_test)


# In[42]:


res.index = test_data.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]

# To download the csv file locally
##res.to_csv('datathon_loan_lr.csv', index=False)         
#files.download('datathon_loan_lr.csv')


# In[43]:


from sklearn.model_selection import StratifiedKFold


# In[44]:


i=1
mean = 0
fmean = 0
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    f1score = f1_score(yvl,pred_test)
    mean += score
    fmean += f1score
    print('#######################')
    print ('accuracy_score',score)
    print('-------------------------')
    print ('F1 Score ',f1score)
    print('#######################')
    i+=1
    pred_test_f = model.predict(test)
    pred = model.predict_proba(xvl)[:,1]

print('----------- Final Mean Score---------------')    
print('###########################################')
print ('\n Mean Validation Accuracy',mean/(i-1))
print ('\n Mean Validation F1 Score',fmean/(i-1))
print('###########################################')
print('-------------------------------------------')


# In[45]:


from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl, pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# In[46]:


res = pd.DataFrame(pred_test_f) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = test_data.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]


# In[47]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']


# In[48]:



sns.distplot(train['Total_Income'])
     


# In[49]:


train['Total_Income_log'] = np.log(train['Total_Income'])
sns.distplot(train['Total_Income_log'])
test['Total_Income_log'] = np.log(test['Total_Income'])


# In[50]:


train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']


# In[51]:


sns.distplot(train['EMI'])


# In[52]:



train['Balance Income'] = train['Total_Income']-(train['EMI']*1000)
test['Balance Income'] = test['Total_Income']-(test['EMI']*1000)
sns.distplot(train['Balance Income'])


# In[53]:



train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)


# In[54]:


X = train.drop('Loan_Status', axis= 1)
y = train.Loan_Status


# In[55]:


i=1
mean = 0
fmean = 0
print('----------- After Features Engineering---------------')    
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    f1score = f1_score(yvl,pred_test)
    mean += score
    fmean += f1score
    print('#######################')
    print ('accuracy_score',score)
    print('-------------------------')
    print ('F1 Score ',f1score)
    print('#######################')
    i+=1
    pred_test_fe = model.predict(test)
    pred = model.predict_proba(xvl)[:,1]


print('----------- Final Mean Score---------------')    
print('###########################################')
print ('\n Mean Validation Accuracy',mean/(i-1))
print ('\n Mean Validation F1 Score',fmean/(i-1))
print('###########################################')
print('-------------------------------------------') 


# In[56]:



from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl, pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
     


# In[57]:


res = pd.DataFrame(pred_test_fe) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = test_data.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]


# In[58]:


from sklearn.tree import  DecisionTreeClassifier
i=1
mean = 0
fmean = 0
print('----------- After Features Engineering---------------')    
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model_tree = DecisionTreeClassifier(random_state=1)
    model_tree.fit(xtr,ytr)
    pred_test=model_tree.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    f1score = f1_score(yvl,pred_test)
    mean += score
    fmean += f1score
    print('#######################')
    print ('accuracy_score',score)
    print('-------------------------')
    print ('F1 Score ',f1score)
    print('#######################')
    i+=1
    pred_test_tree = model_tree.predict(test)
    pred = model_tree.predict_proba(xvl)[:,1]


print('----------- Final Mean Score---------------')    
print('###########################################')
print ('\n Mean Validation Accuracy',mean/(i-1))
print ('\n Mean Validation F1 Score',fmean/(i-1))
print('###########################################')
print('-------------------------------------------')    
     


# In[59]:


res = pd.DataFrame(pred_test_tree) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = test_data.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]


# In[60]:


from tkinter import *


# In[61]:


import joblib


# In[62]:


data = [1.0,4.70048,0,1,1,0,0,1,0,0,1,1,1,0,0,1,0,5416.0,8.597113,0.305556,5110.444]


# In[63]:


input = pd.DataFrame([[1.0,4.70048,0,1,1,0,0,1,0,0,1,1,1,0,0,1,0,5416.0,8.597113,0.305556,5110.444]])
input.shape


# In[64]:


pred_test_tree = model_tree.predict(input)


# In[65]:


test.iloc[0:1,:].shape


# In[66]:


pred_test_fe = model.predict(input)
pred_test_fe


# ## UI

# In[67]:


from tkinter import *


# In[68]:


import joblib


# In[69]:


joblib.dump(model, "model_UI")


# In[70]:


model_UI = joblib.load("model_UI")


# In[71]:


pred_test_fe = model_UI.predict(input)
pred_test_fe


# In[72]:


master = Tk()


# In[ ]:


master.title("Loan Prediction")

def get_entry():
    p1 = e1.get()
    p2 = e2.get()
    p3 = e3.get()
    p4 = e4.get()
    p5 = e5.get()
    p6 = e6.get()
    p7 = e7.get()
    p8 = e8.get()
    p9 = e9.get()
    p10 = e10.get()
    p11 = e11.get()
    p12 = e12.get()
    p13 = e13.get()
    p14 = e14.get()
    p15 = e15.get()
    p16 = e16.get()
    p17 = e17.get()
    p18 = e18.get()
    p19 = e19.get()
    p20 = e20.get()
    p21 = e21.get()
    

    model = joblib.load("model_UI")
    
#     input = [p1,p2,p3,p4,p5,p6,p7,p8,p9]


    input = pd.DataFrame([[1.0,4.70048,0,1,1,0,0,1,0,0,1,+1,1,0,0,1,0,5416.0,8.597113,0.305556,5110.444]])
  #  input = pd.DataFrame([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21]])

    result  = model.predict(input)

    Label(master, text = "Loan Status: ").grid(row=4, column=4)    
    Label(master, text=str(result)).grid(row = 4, column=6)

label = Label(master, text = "Loan prediction", bg = "black", fg="white").grid(row=0, columnspan =2)


Label(master, text = "Credit_History").grid(row=1)
Label(master, text = "LoanAmount_log").grid(row=2)
Label(master, text = "Gender_Female").grid(row=3)
Label(master, text = "Gender_Male").grid(row=4)
Label(master, text = "Married_No").grid(row=5)
Label(master, text = "Married_Yes").grid(row=6)
Label(master, text = "depemdents 1").grid(row=7)
Label(master, text = "dependents 2").grid(row=8)
Label(master, text = "education graduate").grid(row=9)
Label(master, text = "education note graduate").grid(row=10)
Label(master, text = "self employeed").grid(row=11)
Label(master, text = "Gender_Female").grid(row=12)
Label(master, text = "property rural").grid(row=13)
Label(master, text = "property suburban").grid(row=14)
Label(master, text = "property urban").grid(row=15)
Label(master, text = "total income log").grid(row=16)
Label(master, text = "emi").grid(row=17)
Label(master, text = "balance income").grid(row=18)



e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)
e14 = Entry(master)
e15 = Entry(master)
e16 = Entry(master)
e17 = Entry(master)
e18 = Entry(master)
e19 = Entry(master)
e20= Entry(master)
e21 = Entry(master)


e1.grid(row=1, column = 1)
e2.grid(row=2, column = 1)
e3.grid(row=3, column = 1)
e4.grid(row=4, column = 1)
e5.grid(row=5, column = 1)
e6.grid(row=6, column = 1)
e7.grid(row=7, column = 1)
e8.grid(row=8, column = 1)
e9.grid(row=9, column = 1)
e10.grid(row=10, column = 1)
e11.grid(row=11, column = 1)

e12.grid(row=12, column = 1)
e13.grid(row=13, column = 1)
e14.grid(row=14, column = 1)
e15.grid(row=15, column = 1)
e16.grid(row=16, column = 1)
e17.grid(row=17, column = 1)
e18.grid(row=18, column = 1)
e19.grid(row=19, column = 1)
e20.grid(row=20, column = 1)
e21.grid(row=21, column = 1)





Button(master, text="Predict", command = get_entry).grid(row=6, column=4)



mainloop()


# In[ ]:




