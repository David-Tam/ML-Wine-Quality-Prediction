# ML-Wine-Quality-Prediction
This is a project of a graduated level machine learning course that aims to classify good/bad wine base on costum setting (by the user) on the wine quality. The project was originally finished in R (check it out!) but it was improved and written in python language (shown here).

The description and the datasets can be accessed here:

https://archive.ics.uci.edu/dataset/186/wine+quality

Please not that only **white** wine data is used here.

The project will be presented into two parts: 1. Binary classification. 2. Mulitclass Classification
In general, the structure follows:

**1. Data Access and threshold setting**

**2. Data Visualization**
   
**3. Data Preparation**
   
**4. Logistic Regression Approach**   

**5. Random Forest Approach**   

**6. Support Vector Classifier Approach**


## 1. Binary Classification

The feature of this project is returning "good/bad" label for each wine, and the "standard" is set by the user. The reason behind is simply: common people might not be able to quantify the quality for a bottle of wine (like a professonal wine taster!), this might help them to roughly know which wine might fit their need in the future.

One of the feature is "quality" (from 1 to 10, where 10 is perfect) of the wine, we simply set a threshold on this feature to distinguish the label of the wine.

First of all, we type in a number and set the threshold:
![alt text](images/1.png)

Let's say every quality higher than 6 is considered good wine. Here are all the features:
![alt text](images/1a.png)

And their statistical information:
![alt text](images/1b.png)

Now we apply a new feature "judge" to indicate the wine label. This feature is binary (0 means bad wine and 1 is good wine) and depends on the "quality":
![alt text](images/1c.png)

According to the threshold, distributions can be visualized in every features. First of all, let's see how many "bad" and "good" wine via a pie chart:
![alt text](images/1d.png)
![alt text](images/pie_bad_good_wine.png)

The distributions of continuous features can be visualized via histogram. A for-loop is used to serve the purpose:
![alt text](images/1e.png)

For acidity-related features:
![alt text](images/con_fixed_acidity.png)
![alt text](images/con_volatile_acidity.png)
![alt text](images/con_citric_acid.png)
![alt text](images/con_pH.png)

How sweet is the wine?
![alt text](images/con_residual_sugar.png)

How about chlorides?
![alt text](images/con_chlorides.png)

The sulphur dioxide features:
![alt text](images/con_free_sulfur_dioxide.png)
![alt text](images/con_total_sulfur_dioxide.png)
![alt text](images/con_sulphates.png)

The density, meaning the texture of the wine?
![alt text](images/con_density.png)

And the concentration of alcohol:
![alt text](images/con_alcohol.png)

The only caterogrical feature is the quality of the wine:
![alt text](images/1f.png)
![alt text](images/cat_quality.png)

### Data Preparation
![alt text](images/1g.png)

## Logistic Regression Approach
![alt text](images/1h.png)
![alt text](images/1i.png)
![alt text](images/cm_lr1.png)

![alt text](images/1k.png)
![alt text](images/roc_lr1.png)

![alt text](images/1l.png)
![alt text](images/1m.png)
![alt text](images/cm_rf1.png)

![alt text](images/1o.png)
![alt text](images/roc_rf1.png)

![alt text](images/1p.png)
![alt text](images/1q.png)
![alt text](images/cm_svc1.png)

![alt text](images/1s.png)
![alt text](images/roc_svc1.png)

![alt text](images/1t.png)
![alt text](images/cm_all1.png)


![alt text](images/1u.png)
![alt text](images/1v.png)
![alt text](images/roc_all1.png)




## 2. Multi-class Classification

![alt text](images/2a.png)
![alt text](images/2b.png)
![alt text](images/2c.png)
![alt text](images/2d.png)
![alt text](images/2e.png)
![alt text](images/pie_bad_nor_good_wine.png)

![alt text](images/2f.png)
![alt text](images/2g.png)
![alt text](images/2h.png)
![alt text](images/2i.png)
![alt text](images/cm_all2.png)


![alt text](images/2j.png)
