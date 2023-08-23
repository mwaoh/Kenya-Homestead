# Saurabh Gupta v1.0
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 numberOfGuests:float,
                roomType:str,
                maxNights:float,
                minNights:float,
                city:str,
                state:str,
                latitude:float,
                longitude:float,
                numberOfBedsAvailable:float,
                numberOfBedrooms:float,
                allowsChildren:float,
                allowsEvents:float,
                allowsPets:float,
                allowsSmoking:float,
                allowsInfants:float,
                review_count:float,
                amenities:float,
                regular_bed:float,
                relaxing_bed:float,
                kids_bed:float,
                numberofbathroom:float,
                bathroomType:str):
        
        self.numberOfGuests=numberOfGuests
        self.roomType=roomType
        self.maxNights=maxNights
        self.minNights=minNights
        self.city=city
        self.state=state
        self.latitude=latitude
        self.longitude =longitude
        self.numberOfBedsAvailable=numberOfBedsAvailable
        self.numberOfBedrooms=numberOfBedrooms
        self.allowsChildren=allowsChildren
        self.allowsEvents=allowsEvents
        self.allowsPets=allowsPets
        self.allowsSmoking=allowsSmoking
        self.allowsInfants=allowsInfants
        self.review_count=review_count
        self.amenities=amenities
        self.regular_bed=regular_bed
        self.relaxing_bed=relaxing_bed
        self.kids_bed=kids_bed
        self.numberofbathroom=numberofbathroom
        self.bathroomType=bathroomType


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'numberOfGuests':[self.numberOfGuests],
                'roomType':[self.roomType],
                'maxNights':[self.maxNights],
                'minNights':[self.minNights],
                'city':[self.city],
                'state':[self.state],
                'latitude':[self.latitude],
                'longitude':[self.longitude],
                'numberOfBedsAvailable':[self.numberOfBedsAvailable],
                'numberOfBedrooms':[self.numberOfBedrooms],
                'allowsChildren':[self.allowsChildren],
                'allowsEvents':[self.allowsEvents],
                'allowsPets':[self.allowsPets],
                'allowsSmoking':[self.allowsSmoking],
                'allowsInfants':[self.allowsInfants],
                'review_count':[self.review_count],
                'amenities':[self.amenities],
                'regular_bed':[self.regular_bed],
                'relaxing_bed':[self.relaxing_bed],
                'kids_bed':[self.kids_bed],
                'numberofbathroom':[self.numberofbathroom],
                'bathroomType':[self.bathroomType]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
