import pandas as pd
import textdistance as td
import numpy as np

# function used to only extract number part
# of a given input and output its integer
def get_int(price):
    output_price = ''
    price = str(price)
    for letter in price:
        if letter.isnumeric():
            output_price+=str(letter)
        else:
            break
    if output_price == '':
        output_price = '0'
    return int(output_price)

# function used to create a block_key
# based on the price provide
def get_block_price(price):
    block_price = ''
    if price<=200 and price>=0:
        block_price = 'state1 '
        block_price += str(price//10)
    elif price<=1000 and price>300:
        block_price = 'state2 '
        block_price += str(price//100)
    elif price>1000 and price<=100000:
        block_price = 'state3 '
        block_price += str(price//1000)
    elif price>10000 and price<=1000000:
        block_price = 'state4 '
        block_price += str(price//10000)
    else:
        block_price = 'state5 '
        block_price += str(price//100000)
    return block_price

# function used to determine if number in a string
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
 
    
# function uesd to find the brand name from the input string
# the first word withour any number is assumed as 
# the brand namehere
def get_brand(manufacturer):
    res = manufacturer.split()
    output_brand = ''
    if len(res) == 0:
        output_brand = res[0]
    else:
        for word in res:
            if not hasNumbers(word):
                output_brand = word
                break
    return output_brand
                  
# read two csv file which include the products
# need to be blocked
google = pd.read_csv('google.csv',encoding= 'ISO-8859-1')
amazon = pd.read_csv('amazon.csv',encoding= 'ISO-8859-1')

# make a copy and only extract the useful data
amazon_new = amazon.copy()
google_new = google.copy()
amazon_new = amazon_new[['title','idAmazon','price','manufacturer']]
google_new = google_new[['name','id','price','manufacturer']]

# replace the nan in price column 
# by empty string
google_new['price'] = google_new['price'].replace(np.nan, '', regex=True)
amazon_new['price'] = amazon_new['price'].replace(np.nan,'', regex = True)


# create a new column in both data set which records the interge part
# of the product price(would be used for blocking later)
price_list = []
for price in google_new.loc[:,'price']:
    price_list.append(get_int(price))
google_new.loc[:,'Intprice'] = price_list

price_list1 = []
for price in amazon_new.loc[:,'price']:
    price_list1.append(get_int(price))
amazon_new.loc[:,'Intprice'] = price_list1

# replace the nan column to '' in the 
# manufacturer column in google_new
google_new['manufacturer'] = google_new['manufacturer'].replace(np.nan, '', regex=True)



# figure out two block keys for each product in
# google, one is based on product's price
# another is based on its 'manufacturer'
all_block_google = []
for google_id in google_new['id']:
    match_price = []
    match_brand = []
    google_price = google_new.loc[google_new['id'] == google_id, 'Intprice'].iloc[0]
    block_price = get_block_price(google_price)
    # store the product id and its corresponding 
    # block key(based on price) into a list
    match_price.append(block_price)
    match_price.append(google_id)
    all_block_google.append(match_price)
    
    # find the block key based on product's manufacturer 
    # and store in to a list
    if google_new.loc[google_new['id'] == google_id, 'manufacturer'].iloc[0] != '':
        google_manufacturer = google_new.loc[google_new['id'] == google_id, 'manufacturer'].iloc[0]
        brand = get_brand(google_manufacturer)
        match_brand.append(brand)
        match_brand.append(google_id)
        all_block_google.append(match_brand)
    else:
        google_name = google_new.loc[google_new['id'] == google_id, 'name'].iloc[0]
        brand = get_brand(google_name)
        match_brand.append(brand)
        match_brand.append(google_id)
        all_block_google.append(match_brand)
        
# figure out two block keys for each product in
# amazon, one is based on product's price
# another is based on its 'manufacturer'
all_block_amazon = []
for amazon_id in amazon_new['idAmazon']:
    match_price = []
    match_brand = []
    amazon_price = amazon_new.loc[amazon_new['idAmazon'] == amazon_id, 'Intprice'].iloc[0]
    block_price = get_block_price(amazon_price)
    # store the product id and its corresponding 
    # block key into a list
    match_price.append(block_price)
    match_price.append(amazon_id)
    all_block_amazon.append(match_price)
    # find the block key based on product's manufacturer 
    # and store in to a list
    if amazon_new.loc[amazon_new['idAmazon'] == amazon_id, 'manufacturer'].iloc[0] != '':
        amazon_manufacturer = amazon_new.loc[amazon_new['idAmazon'] == amazon_id, 'manufacturer'].iloc[0]
        brand = get_brand(amazon_manufacturer)
        match_brand.append(brand)
        match_brand.append(amazon_id)
        all_block_amazon.append(match_brand)
    else:
        amazon_name = amazon_new.loc[amazon_new['id']== amazon_id,'title'].iloc[0]
        brand = get_brand(google_name)
        match_brand.append(brand)
        match_brand.append(amazon_id)
        all_block_amazon.append(match_brand)
        
# store all the google product ids and its
# block key as dataframe and convert to csv file
# called 'google_blocks.csv'
column_names_google = ['block_key','product_id']
task1b_google = pd.DataFrame(all_block_google, columns = column_names_google,)
task1b_google.to_csv('google_blocks.csv',index = False)

# store all the amazon product ids and its
# block key as dataframe and convert to csv file
# called  'amazon_blocks.csv'
column_names_amazon = ['block_key','product_id']
task1b_amazon = pd.DataFrame(all_block_amazon, columns = column_names_amazon,)
task1b_amazon.to_csv('amazon_blocks.csv',index = False)