# =============================================================================
# IMPORT PACKAGES
# =============================================================================
import pandas as pd
import numpy as np
import datetime
import math


""" IMPORT FILES, KEEP IMPORTANT INFORMATION AND RENAME COLUMNS """

summer = pd.read_excel(r'./Data/Summer Price Data.xlsx')

winter1 = pd.read_excel(r'./Data/Winter Price Data.xlsx',
                        sheet_name='Trabajo',
                        skiprows=5,
                        header=1)
winter1 = winter1.drop(columns=['Unnamed: 0']) 
winter1 = winter1.iloc[:2964,:] # Drop empty rows at the end


winter2 = pd.read_excel(r'./Data/Winter Price Data.xlsx',
                        sheet_name='Hoja1',
                        skiprows=7,
                        header=1)
winter2 = winter2.drop(['C_GESTION', 
                          'PRODUCTO_GRUPO_COMPRADORES', 
                          'PRODUCTO_PRECIO_VENTA', 
                          'Stock Fisico Final'],
                         axis = 1)
winter2 = winter2.iloc[:32760,:] # Drop empty rows at the end
winter2 = winter2.rename(columns = {'PRODUCTO_CODIGO':'Reference',
                                    'A_SECCION':'Section',
                                    'A_ESTACIONALIDAD':'Stationality',
                                    'A_USO':'Use'})

winter3 = pd.read_excel(r'./Data/Winter Price Data.xlsx',
                        sheet_name='Promo')
winter3 = winter3.drop(['Unnamed: 3'], 
                       axis = 1)
winter3 = winter3.rename(columns = {'Ref 365':'Reference', 
                                    'fecha inicio':'Initial_Date_3', 
                                    'fecha fin':'Finish_Date_3'})

winter4 = pd.read_excel(r'./Data/Winter Price Data.xlsx',
                        sheet_name='Hoja2')
winter4 = winter4.rename(columns = {'Producto':'Reference', 
                                    'Precio de descuento': 'Discount_Price'})

sales = pd.read_excel(r'./Data/Sales data 2019.xlsx')
sales = sales.drop(columns=['Unnamed: 0'])
sales = sales.drop(['Referencia'], axis=1) # Wrong references
sales = sales.rename(columns={'date':'Sales_Date',
                              'product_id':'Product_ID',
                              'sales':'Sales_Quantity'})



families = pd.read_excel(r'./Data/families 2019.xlsx')
families = families.drop(columns=['Unnamed: 0'])

prod_ref = pd.read_csv(r'./Data/Products and Refs.csv')
prod_ref = prod_ref.drop(columns=['Unnamed: 0'])
prod_ref = prod_ref.rename(columns={'referencia':'Reference',
                                    'product_id':'Product_ID'})





""" ADD CORRECT PRODUCT REFERENCES TO SALES """

sales_complete = pd.merge(sales,
                          prod_ref,
                          how='left',
                          left_on='Product_ID',
                          right_on='Product_ID')



""" WINTER DATASET CLEANING. WINTER1 IS THE VARIABLE WITH THE IMPORTANT INFORMATION """
# In this dataframe we have the price change for every winter reference throughout the season
# If there is not a price in any day, it means that the price has not changed

winter = winter1.copy()
# Delete no relevant columns
winter = winter.drop(['Comprador', 
                      'Cod Gestion Actual 12.09', 
                      'Precio ACTUAL', 
                      'Cambios 12/09 ', 
                      'con dto', 
                      'USO',
                      'Cod Gestion Actual 12.12',
                      'promo marketing', 
                      'cambio aifora y compradores 30.01',
                      '26.02 (no aifora)', 
                      '28.02(no aifora)', 
                      'Unnamed: 45', 
                      'Unnamed: 46', 
                      'Unnamed: 47', 
                      'FECHA ÚLTIMO DTO', 
                      'precio + bajo / ref con dto',
                      'Quitamos dto pk  pasan a PRP o BC 12/12', 
                      'quitamos dto enero 2020', 
                      'dto en % febrero', 
                      'check', 
                      '% dto actual', 
                      'xxx en dynamics', 
                      'check preco actual', 
                      'igual check', 
                      'STOCK'], 
                       axis = 1)

# Renaming columns
winter = winter.rename(columns = {'Referencia':'Reference', 
                                  '# Seccion':'Section', 
                                  'Estacionalidad':'Category', 
                                  'PVI':'Internal_Selling_Price',
                                  'Cambios 01/10 ':'01_10_2019',
                                  'Cambios aifora 10/10': '10_10_2019', 
                                  'Cambios aifora 17/10': '17_10_2019', 
                                  'cambio 24/10 aifora': '24_10_2019_1',
                                  'cambio 24/10 SP Jon/Iker': '24_10_2019_2', 
                                  'cambio 31/10 aifora': '31_10_2019_1',
                                  'cambio 31/10 SP Jon/Iker': '31_10_2019_2',
                                  'cambio 7/11 aifora': '07_11_2019_1',
                                  'cambio 7/11 SP Jon/Iker': '07_11_2019_2',
                                  'cambio 7/11 Jon/Idoia': '07_11_2019_3',
                                  'cambio 14/11 aifora': '14_11_2019_1',
                                  'cambio 14/11 Sp Jon/iker': '14_11_2019_2',
                                  'cambio compras 14/11': '14_11_2019_3', 
                                  'cambio 21/11 aifora': '21_11_2019_1',
                                  'cambio 21/11 Sp Jon/iker': '21_11_2019_2',
                                  'cambio 21/11 compradores': '21_11_2019_3',
                                  'cambio 05/12 aifora': '05_12_2019_1',
                                  'cambio 05/12 Sp Jon/iker': '05_12_2019_2',
                                  'cambio 12/12 aifora': '12_12_2019_1', 
                                  'cambio 12/12 Sp Jon/iker': '12_12_2019_2',
                                  'cambio 19/12 aifora': '19_12_2019_1', 
                                  'cambio 19/12 Sport+Diverso+Acc+Calcetines': '19_12_2019_2',
                                  'cambio 23/12 para entrar en vigor el 02/01 Aifora': '02_01_2020',
                                  'cambio 10/01 aifora+sport+diverso+calcetines': '10_01_2020',
                                  'cambio aifora 14/01': '14_01_2020',
                                  'cambio 16/01 aifora': '16_01_2020', 
                                  'cambio aifora 23/01': '23_01_2020',
                                  'cambio aifora y compradores 06.02': '06_02_2020',
                                  'cambio aifora y compradores 13.02': '13_02_2020'})


# Delete € symbol
winter = winter.replace('\u20AC','',regex=True)

# Transform in numeric variables comas to points, and convert it to pandas numeric
winter.Internal_Selling_Price = winter.Internal_Selling_Price.str.replace(',','.')
winter.Internal_Selling_Price = pd.to_numeric(winter.Internal_Selling_Price, downcast='float')

# Remove strings in column "19_12_2019_1" such as "PROMO/YES", as it is no relevant info
for i in range(0,len(winter)):
    if type(winter.loc[i,'19_12_2019_1']) != float:
        winter.loc[i,'19_12_2019_1'] = float('nan')
winter['19_12_2019_1'] = pd.to_numeric(winter['19_12_2019_1'], downcast='float')

# Now all the columns for the different dates are numeric
# We have for the same date more than one column, indicating a price change in some products
# Check if all columns refering to the same date have shared or independent values

# 24_10_2019_1 and 24_10_2019_2 do not share any value, so they refer to different products
index_24_10_2019_1 = [i for i in range(0,len(winter)) if math.isnan(winter['24_10_2019_1'][i]) == False]
index_24_10_2019_2 = [i for i in range(0,len(winter)) if math.isnan(winter['24_10_2019_2'][i]) == False]
set(index_24_10_2019_1).intersection(set(index_24_10_2019_2))

# 31_10_2019_1 and 31_10_2019_2 do not share any value, so they refer to different products
index_31_10_2019_1 = [i for i in range(0,len(winter)) if math.isnan(winter['31_10_2019_1'][i]) == False]
index_31_10_2019_2 = [i for i in range(0,len(winter)) if math.isnan(winter['31_10_2019_2'][i]) == False]
set(index_31_10_2019_1).intersection(set(index_31_10_2019_2))

# 07_11_2019_1, 07_11_2019_2 and 07_11_2019_3 do not share any value, so they refer to different products
index_07_11_2019_1 = [i for i in range(0,len(winter)) if math.isnan(winter['07_11_2019_1'][i]) == False]
index_07_11_2019_2 = [i for i in range(0,len(winter)) if math.isnan(winter['07_11_2019_2'][i]) == False]
index_07_11_2019_3 = [i for i in range(0,len(winter)) if math.isnan(winter['07_11_2019_3'][i]) == False]
set(index_07_11_2019_1).intersection(set(index_07_11_2019_2))
set(index_07_11_2019_1).intersection(set(index_07_11_2019_3))

# 14_11_2019_1, 14_11_2019_2 and 14_11_2019_3 do not share any value, so they refer to different products
index_14_11_2019_1 = [i for i in range(0,len(winter)) if math.isnan(winter['14_11_2019_1'][i]) == False]
index_14_11_2019_2 = [i for i in range(0,len(winter)) if math.isnan(winter['14_11_2019_2'][i]) == False]
index_14_11_2019_3 = [i for i in range(0,len(winter)) if math.isnan(winter['14_11_2019_3'][i]) == False]
set(index_14_11_2019_1).intersection(set(index_14_11_2019_2))
set(index_14_11_2019_1).intersection(set(index_14_11_2019_3))

# 21_11_2019_1, 21_11_2019_2 and 21_11_2019_3 do not share any value, so they refer to different products
index_21_11_2019_1 = [i for i in range(0,len(winter)) if math.isnan(winter['21_11_2019_1'][i]) == False]
index_21_11_2019_2 = [i for i in range(0,len(winter)) if math.isnan(winter['21_11_2019_2'][i]) == False]
index_21_11_2019_3 = [i for i in range(0,len(winter)) if math.isnan(winter['21_11_2019_3'][i]) == False]
set(index_21_11_2019_1).intersection(set(index_21_11_2019_2))
set(index_21_11_2019_1).intersection(set(index_21_11_2019_3))

# 05_12_2019_1 and 05_12_2019_2 do not share any value, so they refer to different products
index_05_12_2019_1 = [i for i in range(0,len(winter)) if math.isnan(winter['05_12_2019_1'][i]) == False]
index_05_12_2019_2 = [i for i in range(0,len(winter)) if math.isnan(winter['05_12_2019_2'][i]) == False]
set(index_05_12_2019_1).intersection(set(index_05_12_2019_2))

# 12_12_2019_1 and 12_12_2019_2 do not share any value, so they refer to different products
index_12_12_2019_1 = [i for i in range(0,len(winter)) if math.isnan(winter['12_12_2019_1'][i]) == False]
index_12_12_2019_2 = [i for i in range(0,len(winter)) if math.isnan(winter['12_12_2019_2'][i]) == False]
set(index_12_12_2019_1).intersection(set(index_12_12_2019_2))

# 19_12_2019_1 and 19_12_2019_2 do not share any value, so they refer to different products
index_19_12_2019_1 = [i for i in range(0,len(winter)) if math.isnan(winter['19_12_2019_1'][i]) == False]
index_19_12_2019_2 = [i for i in range(0,len(winter)) if math.isnan(winter['19_12_2019_2'][i]) == False]
set(index_19_12_2019_1).intersection(set(index_19_12_2019_2))


# As we have the price change for the products in different columns for some dates, 
# we copy to the first column all the values of the other columns

# 24_10_2019
for i in range(0,len(winter)):
    if math.isnan(winter['24_10_2019_1'][i]):
        winter.loc[i,'24_10_2019_1'] = winter.loc[i,'24_10_2019_2']

# 31_10_2019
for i in range(0,len(winter)):
    if math.isnan(winter['31_10_2019_1'][i]):
        winter.loc[i,'31_10_2019_1'] = winter.loc[i,'31_10_2019_2']
        
# 07_11_2019     
for i in range(0,len(winter)):
    if math.isnan(winter['07_11_2019_1'][i]):
        winter.loc[i,'07_11_2019_1'] = winter.loc[i,'07_11_2019_2']
for i in range(0,len(winter)):
    if math.isnan(winter['07_11_2019_1'][i]):
        winter.loc[i,'07_11_2019_1'] = winter.loc[i,'07_11_2019_3']

# 14_11_2019
for i in range(0,len(winter)):
    if math.isnan(winter['14_11_2019_1'][i]):
        winter.loc[i,'14_11_2019_1'] = winter.loc[i,'14_11_2019_2']
for i in range(0,len(winter)):
    if math.isnan(winter['14_11_2019_1'][i]):
        winter.loc[i,'14_11_2019_1'] = winter.loc[i,'14_11_2019_3']
        
# 21_11_2019      
for i in range(0,len(winter)):
    if math.isnan(winter['21_11_2019_1'][i]):
        winter.loc[i,'21_11_2019_1'] = winter.loc[i,'21_11_2019_2']
for i in range(0,len(winter)):
    if math.isnan(winter['21_11_2019_1'][i]):
        winter.loc[i,'21_11_2019_1'] = winter.loc[i,'21_11_2019_3']

# 05_12_2019
for i in range(0,len(winter)):
    if math.isnan(winter['05_12_2019_1'][i]):
        winter.loc[i,'05_12_2019_1'] = winter.loc[i,'05_12_2019_2']        
        
# 12_12_2019
for i in range(0,len(winter)):
    if math.isnan(winter['12_12_2019_1'][i]):
        winter.loc[i,'12_12_2019_1'] = winter.loc[i,'12_12_2019_2']

# 19_12_2019     
for i in range(0,len(winter)):
    if math.isnan(winter['19_12_2019_1'][i]):
        winter.loc[i,'19_12_2019_1'] = winter.loc[i,'19_12_2019_2']


# Remove the duplicated columns with the values already extracted 
winter = winter.drop(['24_10_2019_2',
                      '31_10_2019_2',
                      '07_11_2019_2',
                      '07_11_2019_3',
                      '14_11_2019_2', 
                      '14_11_2019_3',
                      '21_11_2019_2', 
                      '21_11_2019_3',
                      '05_12_2019_2',
                      '12_12_2019_2',
                      '19_12_2019_2'],
                      axis=1)
                      
# Rename columns already having the complete information
winter = winter.rename(columns={'24_10_2019_1':'24_10_2019',
                                '31_10_2019_1':'31_10_2019',
                                '07_11_2019_1':'07_11_2019',
                                '14_11_2019_1':'14_11_2019', 
                                '21_11_2019_1':'21_11_2019', 
                                '05_12_2019_1':'05_12_2019',
                                '12_12_2019_1':'12_12_2019',
                                '19_12_2019_1':'19_12_2019'})
 

# Rename columns values of Stationality and Section to numeric values
winter.Category.replace(to_replace=['1 - ZAPATOS',
                                    '4 - BOTINES',
                                    '5 - BOTAS',
                                    '3 - SANDALIAS',
                                    '2 - LONAS/ALPARGATAS'],
                        value=[1,4,5,3,2],
                        inplace=True)            

winter.Section.replace(to_replace=['2 - MUJER',
                                   '7 - ACCESORIOS',
                                   '1 - HOMBRE',
                                   '5 - DIVERSO',
                                   '4 - NIÑA',
                                   '6 - MARROQUINERIA',
                                   '3 - NIÑO',
                                   '8 - CALCETINES'],
                       value=[2,7,1,5,4,6,3,8],
                       inplace=True)    

# We add the price of each item over time
columns_discounts_winter = list(winter.columns[3:22])

for i in range(1,len(columns_discounts_winter)):
    for j in range(0,len(winter)):     
        if math.isnan(winter[columns_discounts_winter[i]][j]):
            winter[columns_discounts_winter[i]][j] = winter[columns_discounts_winter[i-1]][j]

 



""" SUMMER DATASET CLEANING """
# In this dataframe we have the price change for every summer reference throughout the season
# If there is not a price in any day, it means that the price has not changed

# Drop not necessary columns
summer = summer.drop(['COD GST OK', 
                      'COD GST',
                      'MARCA',
                      'USO',
                      'PUNTO ROJO zapatos',
                      'Precio Actual-NO CAMBIAR DE LUGAR ',
                      'STATUS-NO CAMBIAR DE LUGAR ',
                      'PRECIO  ANTES DE BORRAR NIÑO Y NIÑA',
                      'STATUS ANTES DE BORRAR NIÑO Y NIÑA',
                      'rectificación 13.8 quitando precios rebajas 12.08 y 05.08',
                      'Ingleses 3,99€ MAITE 26/07', 
                      'Rebajas 29.07.1',
                      'Rebajas 05.08.1',
                      'Promo  w11', 
                      'Precio w15', 
                      'Promo w19', 
                      'Promo w22',
                      'Rebajas a 0,99€  05.09',
                      'Rebajas Rafa Sandalias 19.09'], 
                     axis = 1)

# Rename columns
summer = summer.rename(columns = {'Referencia':'Reference',
                                  'Saitee': 'Category',
                                  'Seccion': 'Section',
                                  'PVI': 'Internal_Selling_Price',
                                  'Rebajas 21.06': '21_06_2019',
                                  'Rebajas 28.06': '28_06_2019', 
                                  'Rebajas 03.07': '03_07_2019', 
                                  'Rebajas 09.07': '09_07_2019',
                                  'Rebajas 16.07': '16_07_2019', 
                                  'Rebajas 24.07': '24_07_2019',
                                  'Rebajas 29.07': '29_07_2019',
                                  'Rebajas 05.08': '05_08_2019',
                                  'Rebajas 12.08': '12_08_2019',
                                  'Rebajas 19.08': '19_08_2019',
                                  'Rebajas 26.08': '26_08_2019'})

# We replace the €, fill NaN with 0 and replace the string '-' with 0 as well
summer =summer.replace('\u20AC','',regex=True)
summer.fillna(0.0, inplace = True)
summer.replace('-', 0.0, inplace = True)

# Delete four products with PVI = 0, as this is wrong information
summer = summer.drop([13820,13819,10120,10476])
summer = summer.reset_index(drop=True)

# Rename columns values of Category and Section to numeric values
summer.Category.replace(to_replace=['ZAPATOS',
                                    'SANDALIA',
                                    'LONAS',
                                    '0',
                                    'BOTINES'],
                            value=[1,3,2,4,5],
                            inplace=True)  
        

summer.Section.replace(to_replace=['MUJER',
                                   'Mujer',
                                   'ACCS',
                                   'HOMBRE',
                                   'Hombre',
                                   'DIVERSO',
                                   'Diverso',
                                   'NIÑA',
                                   'Niña',
                                   'niña',
                                   'MARROQ',
                                   'NIÑO',
                                   'Niño',
                                   'niño',
                                   'CALCETINES'],
                       value=[2,2,7,1,1,5,5,4,4,4,6,3,3,3,8],
                       inplace=True)


# We add the price of each item over time
columns_discounts_summer = list(summer.columns[3:15])

for i in range(1,len(columns_discounts_summer)):
    for j in range(0,len(summer)):     
        if summer[columns_discounts_summer[i]][j] == 0:
            summer[columns_discounts_summer[i]][j] = summer[columns_discounts_summer[i-1]][j]



""" MERGE SUMMER AND WINTER TO HAVE ALL THE INFO IN ONE DATAFRAME """
""" CREATE DATAFRAME FOR DISCOUNTS AND PRICES FOR ALL PRODUCTS FOR EVERY DAY """

data_merged = pd.merge(winter,summer,how='outer')
# Some products are atemporal, we keep all the prices throughout the year
data_merged = data_merged.groupby('Reference').max().reset_index()
# Transpose data, to have the references in columns
data_merged = data_merged.transpose()

# Fill NaN values of the dataframe with the original price, meaning that when the product is
# out of the season at which it is offered, in the dataframe it has the original selling price
for i in range(0,len(data_merged.columns)):
    for j in range(4,len(data_merged)):
        if (math.isnan(float(data_merged.iloc[j,i])) == True):
            data_merged.iloc[j,i] = data_merged.iloc[3,i]

# Add initial date of sales being the first of april
original_prices = list((data_merged.iloc[3,:]))
original_prices = pd.Series(original_prices, index = data_merged.columns)
data_merged.loc['01_04_2019'] = original_prices

# Create dataframe of prices for every day
data_prices = data_merged.copy()
data_prices.columns = data_prices.iloc[0,:]

data_prices.drop(['Section',
                  'Internal_Selling_Price',
                  'Category',
                  'Reference'], 
                 axis = 0, inplace=True)

data_prices.insert(0,'Dates',data_prices.index)
data_prices.Dates = data_prices.Dates.apply(lambda x:datetime.datetime.strptime(x, '%d_%m_%Y').date())
data_prices.set_index(data_prices.Dates, inplace= True)
data_prices.rename_axis(None, inplace=True)
data_prices = data_prices.iloc[:,1:]

idx = pd.date_range('04-01-2019', '02-29-2020')
data_prices_full = data_prices.reindex(idx)
data_prices_full.ffill(axis = 0, inplace=True) 


# Create dataframe of discount for every day
data_discounts = data_merged.copy()
data_discounts.columns = data_discounts.iloc[0,:]

data_discounts.drop(['Section',
                     'Category',
                     'Reference'], 
                    axis = 0, inplace=True)

for i in range(0,len(data_discounts.columns)):
    for j in range(1,len(data_discounts)):
        value = (data_discounts.iloc[0,i]-data_discounts.iloc[j,i])/data_discounts.iloc[0,i]
        data_discounts.iloc[j][i] = value
        
data_discounts_full = data_discounts.copy()
data_discounts_full = data_discounts_full.iloc[1:,:]     
        
        
data_discounts_full.insert(0,'Dates',data_discounts_full.index)
data_discounts_full.Dates = data_discounts_full.Dates.apply(lambda x:datetime.datetime.strptime(x, '%d_%m_%Y').date())
data_discounts_full.set_index(data_discounts_full.Dates, inplace= True)
data_discounts_full.rename_axis(None, inplace=True)
data_discounts_full = data_discounts_full.iloc[:,1:]


data_discounts_full = data_discounts_full.reindex(idx)
data_discounts_full.ffill(axis = 0, inplace=True) 



""" DATAFRAME WITH THE INFORMATION FOR EACH REFERENCE """
classif = data_merged.iloc[0:4,:]
classif = classif.transpose()
w = list(winter.Reference)
s = list(summer.Reference)

for i in range(0,len(classif)):  
    if classif.Reference[i] in w:    
        classif.loc[i,'Category'] = int(winter.loc[winter.Reference == classif.Reference[i],:].Category)
        classif.loc[i,'Section'] = int(winter.loc[winter.Reference == classif.Reference[i]].Section)
    elif classif.Reference[i] in s:
        classif.loc[i,'Category'] = int(summer.loc[summer.Reference == classif.Reference[i],:].Category)
        classif.loc[i,'Section'] = int(summer.loc[summer.Reference == classif.Reference[i]].Section)

for i in range(0,len(classif)):
    if classif.Section[i] in [6,7,8]:
        classif.Category[i] = 0
        
# Create column for gender and type of product
classif['Gender'] = 0
classif['Product'] = 0

for i in range(0,len(classif)):
    if classif.Section[i] == 1:
        classif.Gender[i] = '1 - Men'
    elif classif.Section[i] == 2:
        classif.Gender[i] = '2 - Women'
    elif classif.Section[i] == 3:
        classif.Gender[i] = '3 - Boy'
    elif classif.Section[i] == 4:
        classif.Gender[i] = '4 - Girl'
    elif classif.Section[i] == 5:
        classif.Gender[i] = '5 - Unisex'
        
for i in range(0,len(classif)):
    if classif.Category[i] == 1:
        classif.Product[i] = '1 - Classic Shoes'
    elif classif.Category[i] == 2:
        classif.Product[i] = '2 - Sneakers'
    elif classif.Category[i] == 3:
        classif.Product[i] = '3 - Sandals'
    elif classif.Category[i] == 4:
        classif.Product[i] = '4 - Ankle Boots'
    elif classif.Category[i] == 5:
        classif.Product[i] = '5 - Boots'
    elif classif.Section[i] == 6:
        classif.Product[i] = '6 - Leather Goods'
    elif classif.Section[i] == 7:
        classif.Product[i] = '7 - Accesories'
    elif classif.Section[i] == 8:
        classif.Product[i] = '8 - Socks'


# Create column for product season
classif['Season'] = 0
for i in range(0,len(classif)):
    if (classif.Reference[i] in w) and (classif.Reference[i] in s):
        classif.Season[i] = 'Atemporal'
    elif classif.Reference[i] in w:
        classif.Season[i] = 'Winter'
    else:
        classif.Season[i] = 'Summer'
        
classif.drop(columns=['Section','Category','Internal_Selling_Price'],inplace=True)
classif.set_axis(classif.Reference,inplace=True)


""" DELETE WRONG REFERENCES """
# Negative discounts

classif.drop(['18122CSN3SP0E77', 
              '18122CSN3SP0N03',
              '18122CSN3SP0O28',
              '18122CSN3SP0P30',
              '19122VEP3YC2Z52',
              '999400000000308'],
             inplace=True)
classif = classif.reset_index(drop=True)

data_discounts_full.drop(columns=['18122CSN3SP0E77',
                                  '18122CSN3SP0N03',
                                  '18122CSN3SP0O28',
                                  '18122CSN3SP0P30',
                                  '19122VEP3YC2Z52',
                                  '999400000000308'],
                         inplace= True)

data_prices_full.drop(columns=['18122CSN3SP0E77',
                               '18122CSN3SP0N03',
                               '18122CSN3SP0O28',
                               '18122CSN3SP0P30',
                               '19122VEP3YC2Z52',
                               '999400000000308'],
                      inplace= True)

sales_complete = sales_complete.drop(sales_complete[sales_complete.Reference.isin(['18122CSN3SP0E77',
                                                                                   '18122CSN3SP0N03',
                                                                                   '18122CSN3SP0O28',
                                                                                   '18122CSN3SP0P30',
                                                                                   '19122VEP3YC2Z52',
                                                                                   '999400000000308'])].index)



""" EXPORT FILES """
classif.to_csv('./V3/EXCELFILES/Classification.csv')
prod_ref.to_csv('./V3/EXCELFILES/Reference_ProductID.csv')
sales_complete.to_csv('./V3/EXCELFILES/Sales.csv')
winter.to_csv('./V3/EXCELFILES/winter_clean.csv')
summer.to_csv('./V3/EXCELFILES/summer_clean.csv')
data_prices_full.to_csv('./V3/EXCELFILES/data_prices_full.csv')
data_discounts_full.to_csv('./V3/EXCELFILES/data_discounts_full.csv')