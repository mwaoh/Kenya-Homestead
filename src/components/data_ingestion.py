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
            
            #removing some columns that are irrelevant to the project goals:
            df.drop(['hostId','memberSince','url','name','address','numberOfLanguages',], axis=1,inplace = True)

            # Drop columns with more than 25% missing values
            def drop_columns(df, threshold):
                for col in df.columns:
                    if df[col].isnull().sum() > threshold:
                        df.drop(col, axis=1, inplace=True)
                return df

            df= drop_columns(df, 0.25*len(df))


            #grouping the room complimentaries to one column named + amenities 
            amenities_list = ['Washer', 'Shampoo', 'Hair dryer', 'Air conditioning', 'Private entrance']
            df['amenities'] = 0
            for amenity in amenities_list:
                df['amenities'] = df['amenities'] | df[amenity]

            #droping complimentaries columns:
            df.drop(amenities_list, axis=1,inplace = True)

            beds = ['double_bed', 'floor_mattress', 'single_bed', 'queen_bed', 'couch', 
            'king_bed', 'air_mattress', 'sofa_bed', 'small_double_bed', 
            'bunk_bed', 'toddler_bed', 'crib', 'hammock', 'water_bed']

            regular_bed = ['double_bed', 'floor_mattress', 'single_bed', 'queen_bed', 'king_bed']
            relaxing_bed = ['couch', 'air_mattress', 'sofa_bed', 'small_double_bed', 'bunk_bed', 'hammock', 'water_bed']
            kids_bed = ['toddler_bed', 'crib']

            # Create columns for each category
            df['regular_bed'] = 0
            df['relaxing_bed'] = 0
            df['kids_bed'] = 0

            #iterating through the list and categorising
            for bed_type in beds:
                if bed_type in regular_bed:
                    df['regular_bed'] = df['regular_bed'] | df[bed_type].astype(int)
                elif bed_type in relaxing_bed:
                    df['relaxing_bed'] = df['relaxing_bed'] | df[bed_type].astype(int)
                elif bed_type in kids_bed:
                    df['kids_bed'] = df['kids_bed'] | df[bed_type].astype(int)

            df.drop(beds, axis=1,inplace = True)

            # Bedrooms columns
            df['numberOfBedrooms'].fillna(1, inplace=True) #mean is 1


            #checking median of the column:
            dff = df[df['numberOfBedrooms'] != "Studio"]

            #distribution of values in the column is relatively normal ; median
            bedroom_median = dff.numberOfBedrooms.median()

            #replace Studio with median of the value
            df.loc[df['numberOfBedrooms'] == "Studio", "numberOfBedrooms"] = bedroom_median

            #changing the column dtype to float
            df['numberOfBedrooms'] = df['numberOfBedrooms'].astype(float)

            #fill null value with median
            df['numberOfBedsAvailable'].fillna(df['numberOfBedsAvailable'].median(), inplace=True)

            df["city"] = df["city"].replace("内罗毕", "Nairobi")
            df['city'].fillna(df['city'].mode()[0], inplace=True)
            df["state"] = df["state"].replace("内罗毕特区", "Nairobi County")
            df['state'].fillna(df['state'].mode()[0], inplace=True)
            df["localizedCity"] = df["localizedCity"].replace("内罗毕", "Nairobi")
            df['localizedCity'].fillna(df['localizedCity'].mode()[0], inplace=True)

            df['Review Count'].fillna(df['Review Count'].median(),inplace= True)
            df.drop(["localizedCheckInTimeWindow"], axis=1,inplace = True)

            #price anlaysis:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

            df['price'].fillna(df['price'].median(), inplace=True)

            df['numberofbathroom']=df.bathroomLabel.str.extract('(\d+)')

            value = df['numberofbathroom'].mode()[0] 

            # Fill NaN values in 'bathroomLabel' column with the mode value
            df['numberofbathroom'] = df['numberofbathroom'].fillna(value)
            df['numberofbathroom'] = df['numberofbathroom'].astype(float)
            
            df['bathroomLabel'].fillna(df['bathroomLabel'].mode()[0], inplace=True)
            df['bathroomType'] = df['bathroomLabel'].str.lower()

            # Classifying  as -  'private', 'shared', or 'unknown'
            df.loc[df['bathroomType'].str.contains('private'), 'bathroomType'] = 'private'
            df.loc[df['bathroomType'].str.contains('shared'), 'bathroomType'] = 'shared'
            df.loc[~df['bathroomType'].str.contains('private|shared'), 'bathroomType'] = 'unknown'

            df.drop('bathroomLabel', axis=1,inplace = True)
            df.drop('localizedCity', axis=1,inplace = True)
            
            #convert all boolean column(true or False ) to numerical

            for column in df.columns:
                if df[column].dtype in [bool]:
                    # Convert boolean columns to numeric (1 or 0)
                    df[column] = df[column].astype(int)
                    
            mapping = {
                'Private room in bed and breakfast': 'Private room',
                'Private room in earthen home': 'Private room', 
                'Shared room in rental unit': 'Shared room',
                'Private room in home': 'Private room',
                'Entire condo': 'Entire unit',
                'Private room in rental unit': 'Private room',
                'Entire rental unit': 'Entire unit',
                'Private room in guest suite': 'Private room',
                'Entire guesthouse': 'Entire unit',
                'Entire loft': 'Entire unit',
                'Shared room in home': 'Shared room',
                'Entire home': 'Entire unit',
                'Entire guest suite': 'Entire unit',
                'Entire vacation home': 'Entire unit',
                'Private room in townhouse': 'Private room',
                'Barn': 'Other',
                'Entire serviced apartment': 'Entire unit',
                'Entire bungalow': 'Entire unit',
                'Tiny home': 'Other',
                'Private room in serviced apartment': 'Private room',
                'Entire villa': 'Entire unit',
                'Private room in condo': 'Private room',
                'Room in hotel': 'Other',
                'Private room in bungalow': 'Private room',
                'Private room in casa particular': 'Private room',
                'Room in boutique hotel': 'Other',
                'Private room in guesthouse': 'Private room',
                'Private room in cottage': 'Private room',
                'Room in bed and breakfast': 'Other',
                'Private room in farm stay': 'Private room',
                'Entire cottage': 'Entire unit',
                'Private room in loft': 'Private room',
                'Private room in tiny home': 'Private room',
                'Private room in nature lodge': 'Private room',
                'Tent': 'Other',
                'Farm stay': 'Other',
                'Shared room in farm stay': 'Shared room',
                'Island': 'Other',
                'Private room in tent': 'Private room',
                'Entire cabin': 'Entire unit',
                'Room in nature lodge': 'Other',
                'Campsite': 'Other',
                'Entire townhouse': 'Entire unit',
                'Hut': 'Other',
                'Private room in resort': 'Private room',
                'Entire chalet': 'Entire unit',
                'Shipping container': 'Other',
                'Treehouse': 'Other',
                'Private room in camper/rv': 'Private room',
                'Room in resort': 'Other',
                'Entire place': 'Entire unit',
                'Shared room in hotel': 'Shared room',
                'Private room in villa': 'Private room',
                'Private room': 'Private room',
                'Room in serviced apartment': 'Other',
                'Earthen home': 'Other',
                'Shared room in townhouse': 'Shared room',
                'Private room in chalet': 'Private room',
                'Private room in vacation home': 'Private room',
                'Shared room': 'Shared room',
                'Entire bed and breakfast': 'Entire unit',
                'Shared room in hostel': 'Shared room',
                'Private room in treehouse': 'Private room',
                'Private room in hut': 'Private room',
                'Shared room in guesthouse': 'Shared room',
                'Shared room in vacation home': 'Shared room',
                'Private room in holiday park': 'Private room',
                'Tipi': 'Other',
                'Shared room in bed and breakfast': 'Shared room',
                'Shared room in boutique hotel': 'Shared room',
                'Room in aparthotel': 'Other',
                'Casa particular': 'Other',
                'Cave': 'Other',
                'Tower': 'Other',
                'Train': 'Other',
                'Private room in dome': 'Private room',
                'Dome': 'Other',
                'Bus': 'Other',
                'Shared room in tiny home': 'Shared room',
                'Private room in cabin': 'Private room',
                'Private room in island': 'Private room',
                'Shared room in hut': 'Shared room',
                'Shared room in loft': 'Shared room',
                'Shared room in bungalow': 'Shared room',
                'Shared room in condo': 'Shared room',
                'Shared room in serviced apartment': 'Shared room',
                'Castle': 'Other',
                'Boat': 'Other',
                'Lighthouse': 'Other',
                'Entire home/apt': 'Entire unit',
                'Private room in hostel': 'Private room',
                'Shared room in ryokan': 'Shared room'
            }

            # Apply the custom mapping to the 'roomType' column
            df['roomType'] = df['roomType'].map(mapping)

            df.drop('personCapacity', axis=1,inplace = True)

            df.rename(columns = {'Review Count':'review_count'}, inplace = True)

            state_list = ['Nairobi County', 'Wilaya ya Kajiado', 'Kajiado County',
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
            city_list = ['Ngong', 'Nairobi', 'Ruaka', 'Pridelands', 'Nairobi City, Kenya',
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

            for stateindf in df['state']:
                for i in range(len(state_list)):
                    if str(stateindf).lower().find(str(state_list[i]).lower()) != -1:
                        df['state'] = df['state'].replace([stateindf], state_list[i]) 

            for cityindf in df['city']:
                for i in range(len(city_list)):
                    if str(cityindf).lower().find(str(city_list[i]).lower()) != -1:
                        df['city'] = df['city'].replace([cityindf], city_list[i])

            

            '''
            #df=df.replace({True:1,False:0,'Studio':1})
            
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
            '''
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(df.columns)
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