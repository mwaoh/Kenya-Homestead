from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

#def home_page():
#    return render_template('index.html')

#@app.route('/predict',methods=['GET','POST'])

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            bathroomLabel=float(request.form.get('bathroomLabel')),
            Private_entrance = float(request.form.get('Private_entrance')),
            numberOfBedsAvailable = float(request.form.get('numberOfBedsAvailable')),
            numberOfBedrooms = float(request.form.get('numberOfBedrooms')),
            roomType = float(request.form.get('roomType')),
            
            toddler_bed = float(request.form.get('toddler_bed')),            
            crib = float(request.form.get('crib')),
            hammock = float(request.form.get('hammock')),
            water_bed = float(request.form.get('water_bed')),
            allowsChildren = float(request.form.get('allowsChildren')),
            allowsEvents = float(request.form.get('allowsEvents')),
            allowsPets = float(request.form.get('allowsPets')),
            allowsSmoking = float(request.form.get('allowsSmoking')),
            allowsInfants = float(request.form.get('allowsInfants')),
            personCapacity = float(request.form.get('personCapacity')),

            

            Accuracy = float(request.form.get('Accuracy')),
            Check_in = float(request.form.get('Check_in')),
            Cleanliness = float(request.form.get('Cleanliness')),
            Communication = float(request.form.get('Communication')),
            Location = float(request.form.get('Location')),
            Review_Count = float(request.form.get('Review_Count')),
            Value = float(request.form.get('Value')),
            amenities = float(request.form.get('amenities')),
            bed_type = float(request.form.get('bed_type'))
           
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)

