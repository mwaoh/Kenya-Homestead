# Saurabh Gupta v1.0
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['roomType', 'city', 'state', 'bathroomType']
            numerical_cols = ['numberOfGuests', 'maxNights', 'minNights', 'latitude', 'longitude', 'numberOfBedsAvailable', 'numberOfBedrooms', 'allowsChildren', 'allowsEvents', 'allowsPets', 'allowsSmoking', 'allowsInfants', 'review_count', 'amenities', 'regular_bed', 'relaxing_bed', 'kids_bed', 'numberofbathroom']
            logging.info('Data Transformation initiated2')

            # Define the custom ranking for each ordinal variable
            #state_categories = ['Baringo','Bomet','Bungoma','Busia','Elgeyo','Mara','Embu','Garissa','Homa Bay','Isiolo','Kajiado','Kakamega','Kericho','Kiambu','Kilifi','Kirinyaga','Kisii','Kisumu','Kitui','Kwale','Laikipia','Lamu','Machakos','Makueni','Mandera','Marsabit','Meru','Migori','Mombasa','Muranga','Nairobi','Nakuru','Nandi','Narok','Nyamira','Nyandarua','Nyeri','Samburu','Siaya','Taveta','Tana','Tharaka-Nithi','Trans-Nzoia','Turkana','Uasin Gishu','Vihiga','Wajir','West Pokot']
            #city_categories = ['Baragoi','Bondo','Bungoma','Busia','Butere','Dadaab','Diani Beach','Eldoret','Emali','Embu','Garissa','Gede','Gem','Hola','Homa Bay','Isiolo','Kitui','Kibwezi','Kajiado','Kakamega','Kakuma','Kapenguria','Kericho','Keroka','Kiambu','Kilifi','Kisii','Kisumu','Kitale','Lamu','Langata','Litein','Lodwar','Lokichoggio','Londiani','Loyangalani','Machakos','Makindu','Malindi','Mandera','Maralal','Marsabit','Meru','Mombasa','Moyale','Mtwapa','Mumias','Muranga','Mutomo','Nairobi','Naivasha','Nakuru','Namanga','Nanyuki','Naro Moru','Narok','Nyahururu','Nyeri','Ruiru','Siaya','Shimoni','Takaungu','Thika','Ugunja','Vihiga','Voi','Wajir','Watamu','Webuye','Wote','Wundanyi']
            #'roomType_categories', 'city_categories', 'state_categories', '', 'bathroomType_categories'
            
            roomType_categories = ['Private room', 'Shared room', 'Entire unit', 'Other']
            bathroomType_categories = ['shared', 'private', 'unknown']
            state_categories = ['Nairobi County', 'Wilaya ya Kajiado', 'Kajiado County',
                'Kiambu County', 'Nairobi', 'Machakos County', 'Eastern',
                'Central', 'Rift Valley', 'Kenya, Nairobi', 'Narok County',
                'Bomet County', 'Kisii County', 'Nakuru County', 'Homa Bay County',
                'Wilaya ya Narok', 'Nyamira County', 'Kericho County',
                'Wilaya ya Kisii Kati', 'Wilaya ya Nakuru', 'Narok', 'Nakuru',
                'Nyanza', 'Laikipia County', 'Meru County', 'Nyeri County',
                'Wilaya ya Isiolo', 'Samburu County', 'Wilaya ya Laikipia',
                'Laikipia', 'Kirinyaga', 'Nyeri', 'Meru', 'Uasin Gishu County',
                'Elgeyo-Marakwet County', 'Wilaya ya Uasin Gishu', 'Naivasha',
                'Nyahururu', 'Kenya', 'Nakuru ', 'Nyandarua County',
                'Taita-Taveta County', 'Kilimanjaro Region', 'Coast',
                'Mombasa County', 'Kwale', 'Kwale County', 'Wilaya ya Kwale',
                'Ukunda, Kwale County', 'Kilifi County', 'Kwale District',
                'Mombasa', 'kenya', 'Wilaya ya Mombasa', 'coast', 'Kwale ',
                'Diani', 'Kisumu County', 'Kakamega County', 'Annex', 'Kisumu',
                'Kakamega', 'Nandi County', ' Rift Valley', 'Wilaya ya Kisumu',
                'Vihiga County', 'Elgeyo Marakwet', 'Trans-Nzoia County',
                'Uasin-Gishu', 'Tharaka-Nithi County', 'Embu County',
                'Makueni County', 'Kitui County', 'Wilaya ya Makueni',
                'Wilaya ya Kakamega', 'Wilaya ya Vihiga', 'Kilifi',
                'Wilaya ya Kilifi', 'Root Node', 'Galu Beach', 'South coast',
                'Nyali', 'Mtwapa', 'Nyali estate', 'Westlands, Nairobi',
                'Kaijado County', 'Kajiado', 'Langata', 'Kiambu', 'Westlands',
                'Nairobi Area', 'Kikuyu', 'Nyamira', 'Wilaya ya Kiambu',
                'Kajiado North County', 'Kileleshwa', 'West Pokot County',
                'Turkana County', 'Isiolo County', 'Kirinyaga County',
                'Kaunti ya Meru', 'Eastern Region', 'Machakos', 'Muranga County',
                'Kenia', 'Kisaju', 'Nyali estate,', 'Homa Bay', 'Migori County',
                'Bungoma County', 'Busia County', 'Siaya', 'Bungoma',
                'western kenya', 'Mara Region', 'Kendu bay', 'Texas',
                'Nairobi City', 'Nairobi, Kenya', 'Wanyee Cl, Nairobi, Kenya',
                'Ruaka', 'South B', 'Nairobi-Upper hill Area',
                'Wilaya ya Machakos', 'Embakasi East', 'Nairobi ', 'NAIROBI',
                'Coast Province', 'Malindi', 'Provincia costiera', 'kilifi',
                'Distretto di Kilifi', 'KF', 'Watamu', 'Kilifi Province',
                'Kilifi, Watamu']
            city_categories = ['Ngong', 'Nairobi', 'Ruaka', 'Pridelands', 'Nairobi City, Kenya',
                'Ongata Rongai', 'Kiambu', 'Athi River', 'Ruiru', 'Nairobi City',
                'Muthaiga North,', 'Kiambu District', 'Kitengela', 'Kiserian',
                'South', 'New Njiru Town', 'Nairobi - Lavington', 'North',
                'Limuru road', 'Mlolongo', 'Westlands', 'Kajiado',
                'Seganani Masai Mara national reserve', 'Bomet', 'Narok',
                'Highway', 'Entasekera', 'Nakuru', 'Kadongo', 'Mau Narok',
                'Lake Elmenteita', 'Ikonge', 'Litein', 'Naivasha ', 'Keroka',
                'Ololaimutiek Village', 'Naivasha', 'Kongoni', 'Gilgil', 'Kisii',
                'Talek', 'Nyanchwa Hill', 'Narok County', 'Aitong', 'Lolgorien',
                'Maasai Mara', 'Oyugis', 'Sekenani', 'Masai Mara', 'Ewaso Ngiro',
                'SEKENANI', 'Silibwet', 'Nanyuki', 'Nchiru', 'Isiolo', 'Dol Dol',
                'Wamba', 'Meru', 'Rukanga, Sagana', 'East', 'Meru District',
                'Timau', 'Maua', 'Eldoret', 'Iten', 'Elmenteita', 'Lake Naivasha',
                'Njoro', 'Eburru', 'Kasuku', 'Nakuru town', 'Voi', 'Mwatate',
                'Taveta', 'Maungu', 'Wundanyi', 'Same', 'Mombasa', 'Diani Beach',
                'Mtwapa', 'Tiwi', 'DIANI BEACH', 'Kwale', 'Ukunda', 'Nyali Beach',
                'Tiwi Beach', 'Galu Beach', 'Nyali Mombasa', 'Nyali',
                'Diani Beach ', 'Diani', 'Diani Beach Road', 'Kisumu',
                'Isukha ICHINA', 'Kakamega', 'Milimani', 'Kapsabet', 'Kisumu City',
                'Gisambai', 'Kapseret', 'Kitale', 'Vihiga', 'Soy', 'Malava',
                'Naro Moru, Nanyuki', 'Naro Moru', 'Ol Kalou', 'Rumuruti',
                'Nanyuki ', 'Nyahururu', 'Laikipia', 'Chuka', 'Gatunga', 'Siakago',
                'Chogoria', 'Igoji', 'Machakos', 'ndagani', 'Matuu', 'Wote',
                'Karurumo', 'Matinyani', 'Mutomo', 'Kitui', 'Mtito Andei',
                'Syongila', 'Kibwezi', 'Luanda', 'Chavakali', 'Maragoli', 'Kilifi',
                'Malindi', 'Kikambala', 'Mtwapa, Mombasa', 'Kaloleni',
                'Mida Creek', 'Gede', 'Watamu', 'Malindi - Mambrui', 'Gongoni',
                'Mtwapa Creek', 'Mariakani', 'Msambweni', 'Mambrui',
                'Utange-Mombasa ', 'Waa', 'Mombasa Bamburi Beach', 'Bamburi',
                'Off Diani Beach Road', 'Diane', 'Galu Kinondo Beach', 'Shimoni',
                'Wasini Island', 'Mombasa ', 'Mombasa Kenya, Box 42961-80100',
                'Kikuyu', 'Limuru', 'Kaijado', 'Tigoni', 'Juja', 'Olooloitikosh',
                'Tigoni Dam', 'Kiserian, Rift Valley, KE', 'Karen Nairobi',
                'Karen', ' Mombasa Road', 'Githurai', 'Ngong Hills', 'Karen/Hardy',
                'Embakasi', 'Ngenda', 'Magadi', 'Limuru Town', 'Thika', 'Ndenderu',
                'Ruaka Town', 'Kahawa Sukari', 'Underpass', 'Rironi',
                'Banana Hill', 'Nyamira', 'Tatu City', 'Limuru Town.', 'Mnagei',
                'Makutano', 'Lokichar', 'Mount Kenya', 'Nanyuki - Timau',
                'Archers Post', 'Ruiri Town', 'Malili', 'Kathonzweni', 'Kimana',
                'Merrueshi', 'Sultan Hamud', 'Nkubu', "Murang'a", 'Kutus',
                'Gaichanjiru', 'Gitugi', 'Kabati', 'Tuthu', 'Kagio', 'Sagana',
                'Ndakaini', 'Syokimau', 'Jacaranda Kenia ', 'Langata',
                'Rusinga Islands', 'MIrogi', 'Mbita', 'Kendu Bay', 'Homa Bay',
                'Kagan', 'Homa Bay Town', 'Muhuru', 'Nyangweso', 'Rongo', 'Suneka',
                'Sare', 'Migori', 'Sindo', 'Mfangano Island', 'Bungoma', 'Malaba',
                'Siaya', 'Chwele', 'Shianda', 'Mumias', 'Miendo', 'Webuye',
                'Kisian', 'Gucha', 'Kilimani', 'Batians Lane', 'Nyeri', 'Kerugoya',
                'Kiriani', 'Kiganjo', 'Iria-Ini', 'Karatina', 'Embu', 'Kimunye',
                'Kibugu', 'Kirinyaga District', 'Runyenjes Town', 'Runyenjes',
                'Matayos', 'Ugunja', 'Busia', 'Funyula', 'Tarime', 'Kericho',
                'Longisa', 'Usenge', 'Bondo', 'Kendu bay', 'Rusinga East',
                'Asembo', 'Cedar Hill', 'Kisaju', 'South B',
                'Hurlingham kilimani Nairobi', 'Starehe', 'Muchatha', 'Elgon Road',
                'Kawaida', 'Wangige', 'Athi River ', ' Athi River',
                'Kenyatta Road', 'Isinya', 'Kangundo', 'Nairobi ', 'Tuala',
                'Mwala', 'NAIROBI ', ' Vipingo', 'Mtwapa ', 'Casuarina', 'Vipingo',
                'Kilifi County', 'Watamu ', 'Uyombo', 'Ватаму', 'Takaungu',
                'kilifi creek', 'Kenya', 'Takaungu Creek', 'Kilifi ',
                'Kilifi, Watamu ', 'Mayungu', 'NYANDARUA ', 'Aberdare Range']


            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            # Categorigal Pipeline

            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[roomType_categories,city_categories,state_categories,bathroomType_categories])),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(input_feature_train_df.columns)
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)