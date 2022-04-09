#!/usr/bin/env python
# coding: utf-8

# In[20]:


from flask import Flask


# In[21]:


app = Flask(__name__)


# In[22]:


import joblib


# In[23]:


model_cart = joblib.load('cc_fraud_cart')


# In[24]:


model_rf = joblib.load('cc_fraud_rf')


# In[25]:


model_boost = joblib.load('cc_fraud_boost')


# In[26]:


model_logreg = joblib.load('cc_fraud_logreg')


# In[27]:


from flask import request, render_template


# In[28]:


@app.route("/", methods=["GET","POST"])

def index():
    if request.method == "POST":
        income = float(request.form.get("income"))
        age = float(request.form.get("age"))
        loan = float(request.form.get("loan"))
        print(income)
        print(age)
        print(loan)
        
        res_cart = model_cart.predict([[income, age, loan]])
        res_rf = model_rf.predict([[income, age, loan]])
        res_gb = model_boost.predict([[income, age, loan]])
        res_logreg = model_logreg.predict([[income, age, loan]])
        
        return(render_template("index.html", result_cart=res_cart, result_rf=res_rf, result_gb=res_gb, result_logreg=res_logreg))
    else:
        return(render_template("index.html", result_cart="Loaded", result_rf="Loaded", result_gb="Loaded", result_logreg="Loaded"))
    


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




