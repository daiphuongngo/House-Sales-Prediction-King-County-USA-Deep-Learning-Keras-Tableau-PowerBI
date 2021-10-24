# House Price Prediction - House Sales in King County, USA - Deep Learning (Keras)

Dataset: https://www.kaggle.com/harlfoxem/housesalesprediction 

Author: Phuong Dai Ngo

Github: https://github.com/daiphuongngo

Language and Tool:

- Python

- Tableau

# Overview

This house price prediction in King County uses Keras deep learning package with Tensorflow backend running with GPU support. No major feature engineering has taken place in this project and as a result, the MAE predicted on the Test set is $483,250.

I had applied my basic learned knowledge and experience of Deep Learning using Keras libraries into this project and read this following notebook for reference:

https://www.kaggle.com/ironfrown/deep-learning-house-price-prediction-keras?fbclid=IwAR35xswXVg15TYDP4LmjvBW4KN0cK2Os0WOrqAZFUVgrJlFjaoedLvDGwo0

### A. Preparation

  #### A.1 Load some standard Python libraries

  #### A.2 Load Keras libraries used in this example 

### B. Data Engineering

  #### B.1 Load all data

  #### B.2 Check NaN

  #### B.3 Change features with date-time type to numeric

  #### B.4 Check null

  #### B.5 Label Encoding for Categorical Features

  #### B.6 Plot bar charts

  #### B.7 Drop NaN

### C. Visualization

  #### C.1 Heatmap

  #### C.2 Remove outliers

  #### C.3 Plot 3 charts of 'price' vs all other features

  #### C.4 Plot box plot of 'price' & 'index'

  #### C.5 Plot 3D chart

### D. Scaling data

  #### D.1 Create X, y

  #### D.2 Split the data

  #### D.3 Check unique values

  #### D.4 Function of drawing history chart

  #### D.5 Scale the data

### E. Modelling

  #### E.1 Train Model Function

  #### E.2 Create a Neural Network model

  #### E.3 Apply GridSearchCV on the model

  #### E.4 Best hyper-parameters

  #### E.5 Evaluate the best model

  #### E.6 MAE

  #### E.7 RMSE

  #### E.8 Conclusion

  #### E.9 Test previous model with best hyper-parameters

  #### E.10 Tuning the model

  #### E.11 Plot the tuned history

  #### E.12 Evalute the model after using best hyper-parameters

    ##### E.12.1 Scaled

    ##### E.12.2 Not scaled

### Visualization:

Bathrooms by Average Sales Price

![Bathrooms by Average Sales Price](https://user-images.githubusercontent.com/70437668/138617762-6160be9e-7ad9-433d-952c-164f2b1ab001.jpg)

Bedrooms by Average Sales Price

![Bedrooms by Average Sales Price](https://user-images.githubusercontent.com/70437668/138617766-125889ea-e2c2-4df8-ab76-ddd8698acc52.jpg)

Conditions by Average Sales Price

![Conditions by Average Sales Price](https://user-images.githubusercontent.com/70437668/138617773-20c19bde-c2b9-477d-9e6d-50a91d0a74ab.jpg)

Countplot - Bathrooms

![Countplot - Bathrooms](https://user-images.githubusercontent.com/70437668/138617776-34aa097a-078b-4779-b605-3b21b5482e8c.jpg)

Countplot - Bedrooms

![Countplot - Bedrooms](https://user-images.githubusercontent.com/70437668/138617784-0bd5e8b8-95de-4762-a81e-588d20539ece.jpg)

Countplot - Conditions

![Countplot - Conditions](https://user-images.githubusercontent.com/70437668/138617790-22b0a18c-4cbc-4cb6-a82e-f861e641b77d.jpg)

Countplot - Floors

![Countplot - Floors](https://user-images.githubusercontent.com/70437668/138617794-1cc4ce0a-9ad2-404c-bad1-1a569726b6b4.jpg)




