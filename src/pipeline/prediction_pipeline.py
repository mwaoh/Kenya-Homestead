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
                 state:str,
                 city:str,
                 bathroomLabel:float,
                 Private_entrance:float,
                 numberOfBedsAvailable:float,                
                 numberOfBedrooms:float,
                 roomType:float,
                 toddler_bed:float,
                 crib:float,
                 hammock:float,
                 water_bed:float,
                 allowsChildren:float,
                 allowsEvents:float,
                 allowsPets:float,
                 allowsSmoking:float,
                 allowsInfants:float,
                 personCapacity:float,
                 Accuracy:float,
                 Check_in:float,
                 Cleanliness:float,
                 Communication:float,
                 Location:float,
                 Review_Count:float,
                 Value:float,
                 amenities:float,
                 bed_type:float):
        
        self.state=state
        self.city=city  
        self.bathroomLabel=bathroomLabel
        self.Private_entrance=Private_entrance
        self.numberOfBedsAvailable=numberOfBedsAvailable        
        self.numberOfBedrooms = numberOfBedrooms
        self.roomType = roomType
        self.toddler_bed = toddler_bed
        self.crib = crib
        self.hammock = hammock
        self.water_bed = water_bed
        self.allowsChildren = allowsChildren
        self.allowsEvents = allowsEvents
        self.allowsPets = allowsPets
        self.allowsSmoking = allowsSmoking
        self.allowsInfants = allowsInfants
        self.personCapacity = personCapacity
        self.Accuracy = Accuracy
        self.Check_in = Check_in
        self.Cleanliness = Cleanliness
        self.Communication = Communication
        self.Location = Location
        self.Review_Count = Review_Count
        self.Value = Value
        self.amenities = amenities
        self.bed_type = bed_type

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'state':[self.state],
                'city':[self.city],
                'bathroomLabel':[self.bathroomLabel],
                'Private_entrance':[self.Private_entrance],
                'numberOfBedsAvailable':[self.numberOfBedsAvailable],               
                'numberOfBedrooms':[self.numberOfBedrooms],
                'roomType':[self.roomType],
                'toddler_bed':[self.toddler_bed],
                'crib':[self.crib],

                'hammock':[self.hammock],
                'water_bed':[self.water_bed],
                'allowsChildren':[self.allowsChildren],               
                'allowsEvents':[self.allowsEvents],
                'allowsPets':[self.allowsPets],
            
                'allowsSmoking':[self.allowsSmoking],
                'allowsInfants':[self.allowsInfants],
                'personCapacity':[self.personCapacity],               
                'Accuracy':[self.Accuracy],
                'Check_in':[self.Check_in],

                'Cleanliness':[self.Cleanliness],
                'Communication':[self.Communication],
                'Location':[self.Location],               
                'Review_Count':[self.Review_Count],
                'Value':[self.Value],
                'amenities':[self.amenities],
                'bed_type':[self.bed_type]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
