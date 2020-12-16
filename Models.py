# =============================================================================
# IMPORT PACKAGES
# =============================================================================
import pandas as pd
import numpy as np
import datetime
from scipy.cluster import hierarchy
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from plotnine import *
import sklearn.metrics as sm

# =============================================================================
# IMPORT FILES
# =============================================================================
discounts = pd.read_csv(r'./data_discounts_full.csv')
prices = pd.read_csv(r'./data_prices_full.csv')
classif = pd.read_csv(r'./Classification.csv')
sales = pd.read_csv(r'./Sales.csv')

# Datetime format
discounts.iloc[:,0] = pd.to_datetime(discounts.iloc[:,0])
prices.iloc[:,0] = pd.to_datetime(prices.iloc[:,0])
sales['Sales_Date'] = pd.to_datetime(sales['Sales_Date'])

# =============================================================================
# DATAFRAME PREPARATION - MODELS
# =============================================================================
# Start 7 days before the 21 of June (sales season) according to the uplift model (14 June)
# We assume that before the 21 of June we did not have discounts
discounts = discounts.loc[discounts['Unnamed: 0']>='2019-06-14']
prices = prices.loc[prices['Unnamed: 0']>='2019-06-14']

# Set the date as index
discounts.index = discounts.iloc[:,0]
prices.index = prices.iloc[:,0]

# Remove the first column (Unnamed)
discounts = discounts.iloc[:,1:]
prices = prices.iloc[:,1:]
classif = classif.iloc[:,1:]
sales = sales.iloc[:,1:]

# Round discounts to two decimals
discounts = discounts.applymap(lambda x: round(x,2))

# We drop NaNs in the sales dataframe
sales = sales.dropna()
# Start the 14 of June to 29 of February
sales = sales.loc[sales['Sales_Date']>='2019-06-14']
sales = sales.reset_index(drop=True)

# References that are sold, as we only focus in sold products assuming total stock is total of sales
ref_sold = sales['Reference'].unique()

# Merge with classif file to know which product is each reference
sales_info = pd.merge(sales,
                      classif,
                      how='left',
                      left_on='Reference',
                      right_on='Reference')

# Days in summer: from 14-June (7 days before first discount) to 30-Sep -> 109 days
# Days in winter: from 24-Sep (7 days before first discount) to 29-Feb -> 159 days
# Total days: 260 days

# Information about products that were sold
classif_sold = classif.loc[classif['Reference'].isin(ref_sold)]

# Sales of the five different types of shoes
sales_shoes = sales_info.loc[sales_info['Product'].isin(['1 - Classic Shoes', 
                                                         '2 - Sneakers', 
                                                         '3 - Sandals', 
                                                         '4 - Ankle Boots', 
                                                         '5 - Boots'])]
sales_shoes = sales_shoes.reset_index(drop=True)

# Classification about the five different types of shoes
classif_shoes = classif_sold.loc[classif_sold['Product'].isin(['1 - Classic Shoes', 
                                                         '2 - Sneakers', 
                                                         '3 - Sandals', 
                                                         '4 - Ankle Boots', 
                                                         '5 - Boots'])]
classif_shoes = classif_shoes.reset_index(drop=True)

# References of sold shoes
ref_sold_shoes = sales_shoes['Reference'].unique()

# Apppend a column with the maximum discount of the product
sales_disc_max = []
for i in range(0,len(sales_shoes)):
    d = discounts[sales_shoes['Reference'][i]].max()
    sales_disc_max.append(round(d,2))
sales_shoes['Disc_Max'] = sales_disc_max
    
# Append a column with the discount at the moment of the sale
sales_with_disc = []
for i in range(0,len(sales_shoes)):
    d = discounts[sales_shoes['Reference'][i]][sales_shoes['Sales_Date'][i]]
    sales_with_disc.append(round(d,2))
sales_shoes['Discount'] = sales_with_disc

# Append a column with the original price
sales_prices_or = []
for i in range(0,len(sales_shoes)):
    p = prices[sales_shoes['Reference'][i]][0]
    sales_prices_or.append(round(p,2))
sales_shoes['Original_Price'] = sales_prices_or 
    
# Append a column with the price at the moment of the sale
sales_prices = []
for i in range(0,len(sales_shoes)):
    p = prices[sales_shoes['Reference'][i]][sales_shoes['Sales_Date'][i]]
    sales_prices.append(round(p,2))
sales_shoes['Price'] = sales_prices

# Group by Reference - create a dataframe with the info of the products for the cluster
model_info = sales_shoes.groupby(by=['Reference'], as_index=False).agg({'Sales_Quantity':'sum',
                                                                     'Disc_Max':'mean',
                                                                     'Discount':'mean',
                                                                     'Original_Price':'mean',
                                                                     'Price':'mean'})

# Calculate first and last day that each product has been sold 
first_sale = []
last_sale = []
for i in ref_sold_shoes:
    first = sales_shoes[sales_shoes['Reference'] == i]['Sales_Date'].min()
    last = sales_shoes[sales_shoes['Reference'] == i]['Sales_Date'].max()
    first_sale.append(first)
    last_sale.append(last)
    
disc_days = pd.DataFrame()
disc_days['Reference'] = ref_sold_shoes
disc_days['First_Sale'] = first_sale
disc_days['Last_Sale'] = last_sale   

# Interval total days between the first and the last sale
disc_days['Total_Days_Sales'] = ((disc_days['Last_Sale']-disc_days['First_Sale']).dt.days)

# Calculate how many days the product has been discounted in total
prod_disc_days = sales_shoes.groupby(by=['Reference','Sales_Date'],
                                  as_index=False).agg({'Discount':'sum'})
prod_disc_days = prod_disc_days[prod_disc_days['Discount'] != 0]
prod_disc_days = prod_disc_days.groupby(by='Reference',as_index=False).agg({'Discount':'count'})
prod_disc_days.columns = ['Reference','Total_Days_Discount']

# Add the number of days that the product has been discounted to "disc_days"
disc_days = pd.merge(disc_days,prod_disc_days,how='left',right_on='Reference',left_on='Reference')

# Some products are never discounted
disc_days = disc_days.fillna(0)

# Calculate first day with discount
first_disc = discounts[discounts>0].min()
first_disc = first_disc.reset_index(drop=False)
first_disc.columns=['Reference','First_Disc_Value']
first_disc = first_disc.loc[first_disc['Reference'].isin(ref_sold_shoes)]
first_disc = first_disc.dropna()

first_disc_day = discounts[discounts>0]
first_disc_day_2 = []
for i in first_disc['Reference']:
    first_d = first_disc_day[i].first_valid_index()
    first_disc_day_2.append(first_d)

first_disc['First_Disc_Day'] = first_disc_day_2

#  Add the first discount day and discount value to "disc_days"
disc_days = pd.merge(disc_days,first_disc,how='left',right_on='Reference',left_on='Reference')

# The products that were never discounted are assigned a first discount of 0 the day 01-03-2020
# which is outside of the range we are considering, meaning that never happens
disc_days['First_Disc_Day'] = disc_days['First_Disc_Day'].fillna(datetime.datetime.strptime('01-03-2020', '%d-%m-%Y'))
disc_days['First_Disc_Value'] = disc_days['First_Disc_Value'].fillna(0)

# Merge with classification to know if the product is from summer or winter
disc_days = pd.merge(disc_days, 
                     classif_shoes, 
                     how='left', 
                     left_on='Reference',
                     right_on='Reference')

# Percentage of days that the product has been discounted during its season
# Days in summer: from 14-June (7 days before first discount) to 30-Sep -> 109 days
# Days in winter: from 24-Sep (7 days before first discount) to 29-Feb -> 159 days
# Total days: 260 days
disc_days['Days_Season'] = 0
disc_days.loc[disc_days['Season'] == 'Winter', 'Days_Season'] = 159
disc_days.loc[disc_days['Season'] == 'Summer', 'Days_Season'] = 109
disc_days.loc[disc_days['Season'] == 'Atemporal', 'Days_Season'] = 260

# Percentage of days with discount in its season
disc_days['Perc_Disc_Season'] = disc_days['Total_Days_Discount']/disc_days['Days_Season']

# Calculate if the product is only sold in its season or also in the other season
# We assign a 1 if it is only sold in its seasonality
# We assign a 0 if it is sold in both seasonalities
disc_days['Seasonality'] = 1
for i in range(0,len(disc_days)):
    if ((disc_days['Season'][i] == 'Summer') and\
        (disc_days['Last_Sale'][i] >= datetime.datetime.strptime('01-10-2019', '%d-%m-%Y'))) or\
        ((disc_days['Season'][i] == 'Winter') and\
        (disc_days['First_Sale'][i] < datetime.datetime.strptime('01-10-2019', '%d-%m-%Y'))):
            disc_days['Seasonality'][i] = 0
                 
# Every product is sold at least once before discounting it

# Add the info about discounts to the cluster dataframe
model_info = pd.merge(model_info,disc_days, how='left', left_on='Reference',right_on='Reference')

# Create columns for sales with and without discount
sales_disc = sales_shoes[sales_shoes['Discount'] != 0].\
    loc[:,['Reference','Sales_Quantity']].\
        groupby(by='Reference',as_index=False).\
            agg({'Sales_Quantity':'sum'})
sales_no_disc = sales_shoes[sales_shoes['Discount'] == 0].\
    loc[:,['Reference','Sales_Quantity']].\
        groupby(by='Reference',as_index=False).\
            agg({'Sales_Quantity':'sum'})

# There are product without sales
sales_disc.columns=['Reference','Sales_Disc']
sales_no_disc.columns=['Reference','Sales_No_Disc']

# Add it to the cluster dataframe
model_info = pd.merge(model_info,sales_disc,how='left',right_on='Reference',left_on='Reference')
model_info = pd.merge(model_info,sales_no_disc,how='left',right_on='Reference',left_on='Reference')
model_info = model_info.fillna(0)

# Create column for price categories, transformed to categorical with LabelEncoder
model_info['Price_cat'] = pd.cut(model_info['Original_Price'], bins=[0,20,35,200], labels=[0,1,2])

# Set the references as index to keep them identified and delete reference column
model_info = model_info.set_index(model_info['Reference'])
model_info.drop(columns=['Reference'],inplace=True)


# =============================================================================
# # HIERARCHY CLUSTERING
# =============================================================================

# Keep the interesting columns for the cluster
cluster = model_info.loc[:,['Sales_Quantity', 
                              'Disc_Max', 
                              'Discount', 
                              'Original_Price', 
                              'Price',
                              'Seasonality', 
                              'Sales_Disc', 
                              'Sales_No_Disc',
                              'Price_cat']]


# Normalise the data
data_scaled = preprocessing.normalize(cluster)
data_scaled = pd.DataFrame(data_scaled, columns=cluster.columns)

# Hierarchy cluster
z = hierarchy.linkage(data_scaled, method='ward')
dendrogram = hierarchy.dendrogram(
     z, 
     p = 4,
     leaf_font_size =10, 
     leaf_rotation =90,
     color_threshold = 70,
     above_threshold_color = '#AAAAAA')

# We see three clusters in the dendrogram
labels = hierarchy.fcluster(z,3,criterion='maxclust')
# Add the assigned corresponding cluster to each row
cluster['Cluster'] = labels 
# Add reference column
cluster['Reference'] = cluster.index
cluster = cluster.reset_index(drop=True)

# Add the info for each reference to know what the product is
cluster_details = pd.merge(cluster,classif_shoes,left_on='Reference',right_on='Reference',how='left')
cluster_details = cluster_details.groupby(by='Cluster',as_index=False).mean()




# =============================================================================
# PREDICTIVE MODEL
# =============================================================================

# Info about sales for each product
pred_sales = sales_shoes.copy()
pred_sales = pred_sales.groupby(by=['Reference'], as_index = False).agg({'Sales_Quantity':'sum'})

# Feature engineering. Find variables for the predictive model

# Find the day corresponding to 7 days before the first discount and each of the 7 next days
# 7 days before the first discount the selling price is the original price
# 7 days after the first discount might have different prices each day
pred_model = first_disc.copy()
pred_model['Days_Before'] = pred_model['First_Disc_Day'] - datetime.timedelta(days=7)
pred_model['Days_After_1'] = pred_model['First_Disc_Day'] + datetime.timedelta(days=1)
pred_model['Days_After_2'] = pred_model['First_Disc_Day'] + datetime.timedelta(days=2)
pred_model['Days_After_3'] = pred_model['First_Disc_Day'] + datetime.timedelta(days=3)
pred_model['Days_After_4'] = pred_model['First_Disc_Day'] + datetime.timedelta(days=4)
pred_model['Days_After_5'] = pred_model['First_Disc_Day'] + datetime.timedelta(days=5)
pred_model['Days_After_6'] = pred_model['First_Disc_Day'] + datetime.timedelta(days=6)

# Set the reference as index
pred_model.index = pred_model['Reference']
pred_model.drop(columns=['Reference'],inplace=True)

# Calculate the total sales of the 7 days before the first discount 
# Calculate the total sales the day of the first discount and next six days
sales_bef = []
sales_fir = []
sales_aft_1 = []
sales_aft_2 = []
sales_aft_3 = []
sales_aft_4 = []
sales_aft_5 = []
sales_aft_6 = []

for i in pred_model.index:
    sales_b = pred_sales.loc[pred_sales['Reference'] == i].\
        loc[pred_sales['Sales_Date'] >= pred_model['Days_Before'][i]].\
            loc[pred_sales['Sales_Date'] < pred_model['First_Disc_Day'][i]]\
                ['Sales_Quantity'].sum()
    sales_f = pred_sales.loc[pred_sales['Reference'] == i].\
        loc[pred_sales['Sales_Date'] == pred_model['First_Disc_Day'][i]]['Sales_Quantity'].sum()
    sales_a_1 = pred_sales.loc[pred_sales['Reference'] == i].\
        loc[pred_sales['Sales_Date'] == pred_model['Days_After_1'][i]]['Sales_Quantity'].sum()
    sales_a_2 = pred_sales.loc[pred_sales['Reference'] == i].\
        loc[pred_sales['Sales_Date'] == pred_model['Days_After_2'][i]]['Sales_Quantity'].sum()
    sales_a_3 = pred_sales.loc[pred_sales['Reference'] == i].\
        loc[pred_sales['Sales_Date'] == pred_model['Days_After_3'][i]]['Sales_Quantity'].sum()
    sales_a_4 = pred_sales.loc[pred_sales['Reference'] == i].\
        loc[pred_sales['Sales_Date'] == pred_model['Days_After_4'][i]]['Sales_Quantity'].sum()
    sales_a_5 = pred_sales.loc[pred_sales['Reference'] == i].\
        loc[pred_sales['Sales_Date'] == pred_model['Days_After_5'][i]]['Sales_Quantity'].sum() 
    sales_a_6 = pred_sales.loc[pred_sales['Reference'] == i].\
        loc[pred_sales['Sales_Date'] == pred_model['Days_After_6'][i]]['Sales_Quantity'].sum()
    sales_bef.append(sales_b)
    sales_fir.append(sales_f)   
    sales_aft_1.append(sales_a_1) 
    sales_aft_2.append(sales_a_2)         
    sales_aft_3.append(sales_a_3)         
    sales_aft_4.append(sales_a_4)         
    sales_aft_5.append(sales_a_5)         
    sales_aft_6.append(sales_a_6)         
        
pred_model['Sales_Bef'] = sales_bef
pred_model['Sales_First'] = sales_fir
pred_model['Sales_Aft_1'] = sales_aft_1
pred_model['Sales_Aft_2'] = sales_aft_2
pred_model['Sales_Aft_3'] = sales_aft_3
pred_model['Sales_Aft_4'] = sales_aft_4
pred_model['Sales_Aft_5'] = sales_aft_5
pred_model['Sales_Aft_6'] = sales_aft_6

# Add the value of the original price at which the sellings are done before the day of the first discount
# Add the price for the day of the first discount and the price for each of the next six days
price_or = []
price_fir = []
price_aft_1 = []
price_aft_2 = []
price_aft_3 = []
price_aft_4 = []
price_aft_5 = []
price_aft_6 = []

for i in pred_model.index:
    price_o = prices.loc[:,i][pred_model['Days_Before'][i]]
    price_f = prices.loc[:,i][pred_model['First_Disc_Day'][i]]
    price_a_1 = prices.loc[:,i][pred_model['Days_After_1'][i]]
    price_a_2 = prices.loc[:,i][pred_model['Days_After_2'][i]]
    price_a_3 = prices.loc[:,i][pred_model['Days_After_3'][i]]
    price_a_4 = prices.loc[:,i][pred_model['Days_After_4'][i]]
    price_a_5 = prices.loc[:,i][pred_model['Days_After_5'][i]]
    price_a_6 = prices.loc[:,i][pred_model['Days_After_6'][i]]
    price_or.append(price_o)
    price_fir.append(price_f)
    price_aft_1.append(price_a_1)
    price_aft_2.append(price_a_2)
    price_aft_3.append(price_a_3)
    price_aft_4.append(price_a_4)
    price_aft_5.append(price_a_5)
    price_aft_6.append(price_a_6)

pred_model['Price_Bef'] = price_or
pred_model['Price_First'] = price_fir
pred_model['Price_Aft_1'] = price_aft_1
pred_model['Price_Aft_2'] = price_aft_2
pred_model['Price_Aft_3'] = price_aft_3
pred_model['Price_Aft_4'] = price_aft_4
pred_model['Price_Aft_5'] = price_aft_5
pred_model['Price_Aft_6'] = price_aft_6


# Total of sales including the first day of the discount and the next six days
pred_model['Sales_Aft'] = pred_model['Sales_First'] + pred_model['Sales_Aft_1'] + pred_model['Sales_Aft_2'] + pred_model['Sales_Aft_3'] + pred_model['Sales_Aft_4'] + pred_model['Sales_Aft_5'] + pred_model['Sales_Aft_6']

# Weighted average of the price at which the products have been purchased th first day
# of the discount and the next six days
pred_model['Price_Aft'] = (pred_model['Price_First']*pred_model['Sales_First'] + pred_model['Price_Aft_1']*pred_model['Sales_Aft_1'] + pred_model['Price_Aft_2']*pred_model['Sales_Aft_2'] + pred_model['Price_Aft_3']*pred_model['Sales_Aft_3'] + pred_model['Price_Aft_4']*pred_model['Sales_Aft_4'] + pred_model['Price_Aft_5']*pred_model['Sales_Aft_5'] + pred_model['Price_Aft_6']*pred_model['Sales_Aft_6'])/pred_model['Sales_Aft']

# Some products have not been purchased after the first discount, so NaNs are filled
# with the last available price
pred_model['Price_Aft'] = pred_model['Price_Aft'].fillna(pred_model['Price_Aft_6'])

# Month of the first discount
pred_model['First_Disc_Mon'] = pred_model['First_Disc_Day'].dt.month

# Day of the week of first discount
pred_model['First_Disc_DW'] = pred_model['First_Disc_Day'].dt.day

# Add the info gathered for the cluster
pred_model.reset_index(drop=False,inplace=True)
pred_model = pd.merge(pred_model,
                      cluster_details,
                      how='left',
                      right_on='Reference',
                      left_on='Reference')

# Ratio sales before discount - original price
pred_model['Sales_Bef_OP_Ratio'] = pred_model['Sales_Bef']/pred_model['Original_Price']

# Calculate the uplift
# Ecuation: 1-((Price_Before-Price_After)/(Sales_Before-Sales_After))
pred_model['Uplift'] = 1-((pred_model['Price_Bef']-pred_model['Price_Aft'])/(pred_model['Sales_Bef']-pred_model['Sales_Aft']))

# Replace infinite values with zeros, as it won't be included in target variable
pred_model.replace([np.inf, -np.inf], 0, inplace=True) 

# Target variable creation
pred_model['Target'] = np.where(pred_model['Uplift'] > 1, 1, 0)


# Save dataframes as csv and import them so I do not have to run the whole code
#pred_model.to_csv('Pred_Model.csv')

pred_model = pd.read_csv('Pred_Model.csv')
pred_model.index = pred_model['Reference']
pred_model.drop(columns=['Reference'],inplace=True)
pred_model = pred_model.iloc[:,1:]

pred_model['First_Disc_Day'] = pd.to_datetime(pred_model['First_Disc_Day'])
pred_model['Days_Before'] = pd.to_datetime(pred_model['Days_Before'])
pred_model['Days_After_1'] = pd.to_datetime(pred_model['Days_After_1'])
pred_model['Days_After_2'] = pd.to_datetime(pred_model['Days_After_2'])
pred_model['Days_After_3'] = pd.to_datetime(pred_model['Days_After_3'])
pred_model['Days_After_4'] = pd.to_datetime(pred_model['Days_After_4'])
pred_model['Days_After_5'] = pd.to_datetime(pred_model['Days_After_5'])
pred_model['Days_After_6'] = pd.to_datetime(pred_model['Days_After_6'])

# Specify variables that are going to be used for the model
# Explanatory and target variables

X = pred_model.loc[:,['Original_Price',
                      'Price_cat',
                      'First_Disc_Value',
                      'Season',
                      'Gender', 
                      'Product',
                      'Seasonality',
                      'First_Disc_DW',
                      'Sales_Bef',
                      'Cluster',
                      'Sales_Bef_OP_Ratio',
                      'First_Disc_Mon']]

Y = pred_model['Target']

# LabelEncoder for the categorical variables
le_price_cat = preprocessing.LabelEncoder()
le_season = preprocessing.LabelEncoder()
le_gender = preprocessing.LabelEncoder()
le_product = preprocessing.LabelEncoder()

le_price_cat.fit(X.Price_cat)
le_season.fit(X.Season)
le_gender.fit(X.Gender)
le_product.fit(X.Product)

X.Price_cat = le_price_cat.transform(X.Price_cat)
X.Season = le_season.transform(X.Season)
X.Gender = le_gender.transform(X.Gender)
X.Product = le_product.transform(X.Product)

# Split in train and test
X_train, X_test , Y_train , Y_test = train_test_split (X, 
                                                       Y, 
                                                       test_size = 0.20, 
                                                       random_state = 134)    

# Chosen model after checking different models and parameters with GridSearchCV
model = RandomForestClassifier(class_weight='balanced', 
                               min_samples_leaf=5,
                               min_samples_split=12, 
                               n_estimators=18)
model.fit(X_train, Y_train)
model.score(X_test,Y_test)


# IMPORTANT VARIABLES
feat = model.feature_importances_
dfiv = pd.DataFrame({'Variables':X.columns,'Importance':feat})
ggplot(aes(x='Variables',y="Importance"),dfiv) + geom_bar(stat="identity") + theme(axis_text_x = element_text(angle = 90))


# PREDICTIONS
predictions_proba = model.predict_proba(X_test)
predictions_proba = pd.DataFrame(predictions_proba)
predictions = model.predict(X_test)

# CONFUSION MATRIX
sm.confusion_matrix(Y_test,predictions)
report = sm.classification_report(Y_test,predictions)


# ANALYSE THE CORRECT PREDICTIONS
Y_test = Y_test.reset_index(drop=False)
X = X.reset_index(drop=False)
Y_test['Prediction'] = predictions
Y_test = pd.merge(Y_test,X,how='left',left_on='index',right_on='index')
Y_test['Price_cat'] = le_price_cat.inverse_transform(Y_test['Price_cat'])
Y_test['Season'] = le_season.inverse_transform(Y_test['Season'])
Y_test['Gender'] = le_gender.inverse_transform(Y_test['Gender'])
Y_test['Product'] = le_product.inverse_transform(Y_test['Product'])
Y_test['Correct'] = np.where(Y_test['Prediction'] == Y_test['Target'],1,0)

# ANALYSE THE TRUE POSITIVES
Y_true_pos = Y_test.loc[(Y_test['Target']==1) & (Y_test['Prediction']==1)]

# ANALYSE THE TRUE NEGATIVES
Y_true_neg = Y_test.loc[(Y_test['Target']==0) & (Y_test['Prediction']==0)]

# ANALYSE THE FALSE POSITIVES
Y_false_pos = Y_test.loc[(Y_test['Target']==0) & (Y_test['Prediction']==1)]

# ANALYSE THE FALSE NEGATIVES
Y_false_neg = Y_test.loc[(Y_test['Target']==1) & (Y_test['Prediction']==0)]