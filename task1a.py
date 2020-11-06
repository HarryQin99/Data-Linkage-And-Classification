import pandas as pd
import textdistance as td
from fuzzywuzzy import fuzz
import numpy as np

# function uesd to check is two prices
# provided is 'close' enough.
def check_close_price(price1, price2):
    abs_diff = abs(price1-price2)
    if price1<price2:
        if abs_diff < price1*0.1:
            return 1
    else:
        if abs_diff< price2*0.1:
            return 1
    return 0

# the final score function uesd to calculate the
# final similarity score of the product from google
# and amazon
def final_scoring(simility_scoring, manufacturer_scoring, close_price):
    total_scoring = 0
    total_scoring += simility_scoring
    if manufacturer_scoring > 75:
        total_scoring +=10
    if close_price:
        total_scoring +=5
    return total_scoring

# read two csv file google_small.csv and
# amazon_small.csv first
google = pd.read_csv('google_small.csv',encoding = 'ISO-8859-1')
amazon = pd.read_csv('amazon_small.csv',encoding = 'ISO-8859-1')

# preprocessing for the price column in two
# data sets, avoid error in the comparison later
google['price']=google['price'].astype(float)
amazon['price']=amazon['price'].astype(float)
google['manufacturer'] = google['manufacturer'].replace(np.nan, '', regex=True)

# use a nested loop to fine the match pair
# for each csv file
matches = []
for google_data in google['name']:
    scoring_list = []
    scoring_diction = {}
    match = []
    google_manufacturer = google.loc[google['name'] == google_data,'manufacturer'].iloc[0]
    google_price = google.loc[google['name'] == google_data,'price'].iloc[0]

    for amazon_data in amazon['title']:
        total_scoring = 0
        amazon_manufacturer = ''
        manufacturer_scoring = 0
        if pd.notnull(amazon.loc[amazon['title'] == amazon_data, 'manufacturer'].iloc[0]):
            amazon_manufacturer = amazon.loc[amazon['title'] == amazon_data, 'manufacturer'].iloc[0]
        amazon_price = amazon.loc[amazon['title'] == amazon_data, 'price'].iloc[0]

        # use the similarity function fuzz.token_set_ratio to
        # find how similarity between the product's name in google
        # and the product's title in amazon
        simility_scoring = fuzz.token_set_ratio(google_data,amazon_data)


        # figure out if two product have same manufacturer
        if amazon_manufacturer != '' and google_manufacturer != '':
            manufacturer_scoring = fuzz.token_set_ratio(google_manufacturer,amazon_manufacturer)
        elif amazon_manufacturer != '' and google_manufacturer == '':
            manufacturer_scoring = fuzz.token_set_ratio(google_data,amazon_manufacturer)
        else:
            manufacturer = 0

        # check if the prices of
        # these products are close
        close_price = check_close_price(amazon_price,google_price)

        # use the final_scoring function to calculate a final similarity
        # score of these two products, based on the similarity_scoring,
        # manufacturer_scoring and close_price got before
        total_scoring = final_scoring(simility_scoring, manufacturer_scoring, close_price)

        # determine if these two product are similar base on
        # similarity score they got, threshold is 0.68 here
        if total_scoring > 70:
            scoring_list.append(total_scoring)
            scoring_diction[total_scoring] = amazon_data

    # find the mose highest similarity score
    # and the corresponding amazon product
    if scoring_list != []:
        maxi_scoring = sorted(scoring_list)[-1]
        amazon_data = scoring_diction[maxi_scoring]
        google_id = google.loc[google['name'] == google_data, 'idGoogleBase'].iloc[0]
        amazon_id = amazon.loc[amazon['title'] == amazon_data, 'idAmazon'].iloc[0]
        match.append(google_id)
        match.append(amazon_id)
        # append this match pair into a total matches list
        if match not in matches:
            matches.append(match)

# store the match pairs into a csv file called 'task1a.csv'
column_names = ['idGoogleBase','idAmazon']
task1 = pd.DataFrame(matches, columns = column_names,)
task1.to_csv('task1a.csv',index = False) 
