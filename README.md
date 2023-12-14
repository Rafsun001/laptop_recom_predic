# Laptop Price Prediction & Recommendation
## Content

[**1. Laptop Price Prediction**](#laptop-price-prediction)

[**2. Laptop Recommendation**](#laptop-recommendation)
### Laptop Price Prediction
In this prediction system user need to pass the specification of a laptop and it will predict the price of that laptop. All the laptop information present in the dataset was collected from amazon official site through web scraping. 

#### Data Cleaning 
1. Dropping column
2. Convert columns into correct data type
3. Filling missing values
4. Convert some values of a column into correct form.

#### Creation of Some New Features
1. Creation of  memory and memory type feature from Memory column.
2. Creation of  cpu_model_name and cpu_GHz feature from Cp column.
3. Creation of  screen resolution_x and  screen resolution_y features from ScreenResolution column.
5. Creation of  screen type feature from ScreenResolution column.
6. Creation of touch screen features from ScreenResolution column.
7. Creation of  ppi column using scrn_reslu_y, scrn_reslu_x and Inches columns.
8. Creation of  gpu brand name from Gpu column.
9. Creation of os feature usinng OpSys column.

#### EDA
**Some charts are created:**
1. Pie Chart
2. Bar Chart
3. Scatter Plot

#### Outliers Handling
1. To visualize the outliers box plot and dist plots are used.
2. After detection is done using IQR(Inter Quartile Range) method. 
3. After detecting the outliers some outliers has been removed and some replaced with other value.
 
**How IQR Works?**

The Interquartile Range (IQR) is a statistical measure used to identify outliers in a dataset. It's based on the spread of the middle portion of the data and is calculated as the difference between the third quartile (Q3) and the first quartile (Q1).

*The steps to use IQR to detect outliers are:*

*1. Calculate the IQR:*
Find the first quartile (Q1), which represents the 25th percentile of the data.
Find the third quartile (Q3), which represents the 75th percentile of the data.
Calculate the IQR as IQR = Q3 - Q1.

*2. Define Outliers:*
Any value that falls below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is considered an outlier.

*3. Identify Outliers:*
Check each data point in your dataset against the defined boundaries:
Values below Q1 - 1.5 * IQR are considered outliers on the lower end.
Values above Q3 + 1.5 * IQR are considered outliers on the upper end.

#### Encoding
Encoding is done using label and one-hot encoding method.

**How label encoding works?**

Label encoding, encodes our categorical values in one column. Suppose we have 3 values in a column shirt, pant and watch. What label encoder will do, it will convert shirt into 0,pant into 1, and watch into 2. Here it will not create any new column it will encode in the same column. It means in that column where it will find good there it will use 0, similarly for very good and bad it will use 1 and 2.

**How one-hot encoding works?**

One-hot encoding is a technique used in machine learning to convert categorical data, represented as labels or integers, into a format that can be provided to machine learning algorithms. It's particularly useful when dealing with categorical variables that don't have a numerical relationship between categories.

Suppose you have a categorical feature, such as "Color," with three categories: Red, Green, and Blue

*Step 1:*
Initially, these categories might be encoded as integers, for example:

Red: 0

Green: 1

Blue: 2

*Step 2:*

Each category is represented as a binary vector.
For each category, a vector of length equal to the number of unique categories is created.
Each category's binary vector has all zeros except for a single 1 in the position corresponding to the category index.

For our example:

Red becomes [1, 0, 0]

Green becomes [0, 1, 0]

Blue becomes [0, 0, 1]

#### Model Training
For training XGBoost Regressor is used.

##### **How XGBoost Works?**
XGboost stands for extreme gradient boost. XGboost is an advanced technique of gradient boost. Decision trees are used in both boost and gradient boost models.

**Basic things which make XGboost advance from Gradient boost:**

**Lambda(λ):** It is nothing but a regularization parameter.

**Eta(η):** Eta is the learning rate. Learning rate means at what shift or speed you want changes the predicted value. In xgboost commonly eta value is taken as 0.3 but you can also take between 0.1-1.0

**Similarity score(SS):** SS=(sum of residuals)^2/number of residuals+λ.
When the sign of residuals is opposite then you will get a lower similarity score, it happens because the opposite sign similarity score cancels each other and if not then you will get a higher similarity score. Here lambda is used to control ss. Lambda adds a penalty in ss. This penalty helps to shrink extreme leaf or node weights which can stabilize the model at the cost of introducing bias.

**Gain** = S.S of the branch before split - S.S of the branch after the split

**Gamma(γ):** Gamma is a threshold that defines auto pruning of the tree and controls overfitting. If the gamma value is less than the gain value only then the split will happen in the decision tree otherwise the nodes will not split. So gamma decides that how far the tree will grow. The tree will grow until the value of gain is less than the gamma value. That moment when gain value more than gamma value, that moment growing of tree will stop.

Because XGBoost is an extreme or advanced version of gradient boost so the basic working process will be the same. So before learning about the xgboost working process you must know how gradient boost works. In XGBoost main working process like, find a basic prediction of the main target column by finding the mean of that target column, then find the residuals and then make residuals as the target, then train model, then again get new residuals as prediction, then add new residuals with basic prediction, then again find new residuals, then again train new model everything is same but the difference is how you create the tree, how you add tree predictions residual with basic prediction and what you will get by finding mean of the main target column.
To create a tree first take residuals for the root node. Then use conditions and split nodes like a normal tree. Trees are created in XGBoost normally like how you create in a decision tree. But here in each node, you calculate the similarity score. For one node and its child nodes, you calculate the gain value and you also have a value called gamma. With these values, you can control the overfitting of the decision tree and can get good accuracy. If the gamma value is less than the gain value then only the tree grows or that node can go forward otherwise not. By doing this you perform auto pruning and control overfitting.

**How to add ml prediction value with basic prediction:**
Formula: New prediction=previous prediction+(η*Model residual prediction)
So xgboost also works on residuals and tries to reduce it to get better prediction or accuracy like gradient boost. But here some extra parameters are used like gamma, eta, ss, gain, lambda or do some extra work to perform better from gradient boost.

*Let's see a example:*
This is the data and here dependent column is IQ.

| Age | IQ |
| ------ | ----------- |
| 20 | 38 |
| 15 | 34 |
| 10 | 20 |

Let's find the predicted value and residuals. To get the predicted value to calculate the mean of the dependent variable and here that is 30. To find the residuals subtract the dependent variable with the predicted value.

| Age | IQ | Predicted value | Residual |
| ------ | ------ | ------ | ------ |
| 20 | 38 | 30 | 8 |
| 15 | 34 | 30 | 4 |
| 10 | 20 | 30 | -10 |

*Lets calculate the similarity score:*

First put ƛ as 0

*Formula:* Similarity Score = (S.R2) / (N + ƛ)

Similarity Score(SS) = (-10+4+8)2 / 3+0 = 4/3 = 1.33

Now make a decision tree.

Let's set the tree splitting criteria for Age greater than 10(>10).

![XGBR](https://github.com/Rafsun001/laptop_recom_predic/blob/main/XGBR.png?raw=true )

Now calculate SS and the gain.

For left side leaves the ss:

SS=(-10)^2/ 1+0=100

For right side leaves the ss:

SS=(4+8)^2/ 2+0=72

gain=(100+72)-1.33

Now if the gain is greater than the gamma value only then the splitting will happen of these leaves. For example, let's take the gamma value as 135. So here gain value is greater than gamma so the splitting will happen.

**Now let's see the prediction:**

*Formula:* New prediction=previous prediction+(η*Model residual prediction)

Now put the values in the formula and get the new prediction. Then all those processes will happen again and again until the residuals become zero.
