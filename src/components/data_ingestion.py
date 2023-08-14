# Saurabh Gupta v1.0
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

logging.info('Data Ingestion methods Starts')
## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv(os.path.join('notebooks/datafiles','cleaned_air_bnb_Jul_26.csv'))
            logging.info('Dataset read as pandas Dataframe')
            logging.info('Ingestion of Data is completed')
            df=df.replace({True:1,False:0,'Studio':1})
            
            #g = df.dropna(subset=['state']).drop_duplicates('city').set_index('city')['state']
            #df['new_state'] = df['state'].fillna(df['city'].map(g))

            #g = df.dropna(subset=['city']).drop_duplicates('new_state').set_index('new_state')['city']
            #df['new_city'] = df['city'].fillna(df['new_state'].map(g))

            df['new_roomType'] = df.apply(lambda x: self.addnewfeatures(x['roomType']), axis=1)
            df['bathroomLabel']=df.bathroomLabel.str.extract('(\d+)').astype("float")

            df['amenities'] = df['Washer']+df['Shampoo']+df['Hair dryer']+df['Air conditioning']
            df['bed_type'] = df['double_bed']+df['floor_mattress']+df['single_bed']+df['queen_bed']+df['couch']+df['king_bed']+df['air_mattress']+df['sofa_bed']+df['small_double_bed']+df['bunk_bed']
            
            df = df.drop(columns=['roomType','url','name','address','localizedCheckInTimeWindow','localizedCheckOutTime',
                      'responseTime','localizedCity','hostId','memberSince','numberOfLanguages','numberOfGuests',
                      'maxNights','minNights','latitude','longitude','Washer','Shampoo','Hair dryer','Air conditioning','double_bed','floor_mattress','single_bed','queen_bed','couch','king_bed','air_mattress','sofa_bed','small_double_bed','bunk_bed'],axis=1)
            df.rename(columns = {'Review Count':'Review_Count','Private entrance':'Private_entrance','new_roomType':'roomType','Check-in':'Check_in',}, inplace = True)
            # 'new_city':'city','new_state':'state',
            df = df.dropna(subset = ['price','bathroomLabel'])
            indexPrice = df[ df['price'] >= 200 ].index
            df.drop(indexPrice, inplace=True)

            df['bathroomLabel'].astype("float")
            df['numberOfBedrooms'].astype("float")

            #df = df.drop(columns=['city','state'])
            state_list = ['Baringo','Bomet','Bungoma','Busia','Elgeyo','Mara','Embu','Garissa','Homa Bay','Isiolo','Kajiado','Kakamega','Kericho','Kiambu','Kilifi','Kirinyaga','Kisii','Kisumu','Kitui','Kwale','Laikipia','Lamu','Machakos','Makueni','Mandera','Marsabit','Meru','Migori','Mombasa','Muranga','Nairobi','Nakuru','Nandi','Narok','Nyamira','Nyandarua','Nyeri','Samburu','Siaya','Taveta','Tana','Tharaka-Nithi','Trans-Nzoia','Turkana','Uasin Gishu','Vihiga','Wajir','West Pokot']
            city_list = ['Baragoi','Bondo','Bungoma','Busia','Butere','Dadaab','Diani Beach','Eldoret','Emali','Embu','Garissa','Gede','Gem','Hola','Homa Bay','Isiolo','Kitui','Kibwezi','Kajiado','Kakamega','Kakuma','Kapenguria','Kericho','Keroka','Kiambu','Kilifi','Kisii','Kisumu','Kitale','Lamu','Langata','Litein','Lodwar','Lokichoggio','Londiani','Loyangalani','Machakos','Makindu','Malindi','Mandera','Maralal','Marsabit','Meru','Mombasa','Moyale','Mtwapa','Mumias','Muranga','Mutomo','Nairobi','Naivasha','Nakuru','Namanga','Nanyuki','Naro Moru','Narok','Nyahururu','Nyeri','Ruiru','Siaya','Shimoni','Takaungu','Thika','Ugunja','Vihiga','Voi','Wajir','Watamu','Webuye','Wote','Wundanyi']

            for stateindf in df['state']:
                for i in range(len(state_list)):
                    if str(stateindf).lower().find(str(state_list[i]).lower()) != -1:
                        df['state'] = df['state'].replace([stateindf], state_list[i]) 

            for cityindf in df['city']:
                for i in range(len(city_list)):
                    if str(cityindf).lower().find(str(city_list[i]).lower()) != -1:
                        df['city'] = df['city'].replace([cityindf], city_list[i]) 

            df.loc[~df['city'].isin(city_list), 'city'] = np.nan
            df.loc[~df['state'].isin(state_list), 'state'] = np.nan

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)



    def addnewfeatures(self, x):
        logging.info('Data Ingestion methods Starts')
        try:
            if x.find('Private') > 0:
                #print("private")
                return 0
            elif x.find('Shared') > 0:
                #print("shared")
                return 1
            elif x.find('Entire') > 0:
                #print("Entire")
                return 2
            else:
                return 3
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage Room Type')
            raise CustomException(e,sys)