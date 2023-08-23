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
            numberOfGuests=float(request.form.get('numberOfGuests')),
            roomType=str(request.form.get('roomType')),
            maxNights=float(request.form.get('maxNights')),
            minNights=float(request.form.get('minNights')),
            city=str(request.form.get('city')),
            state=str(request.form.get('state')),
            latitude=float(request.form.get('latitude')),
            longitude=float(request.form.get('longitude')),
            price=float(request.form.get('price')),
            numberOfBedsAvailable=float(request.form.get('numberOfBedsAvailable')),
            numberOfBedrooms=float(request.form.get('numberOfBedrooms')),
            allowsChildren=float(request.form.get('allowsChildren')),
            allowsEvents=float(request.form.get('allowsEvents')),
            allowsPets=float(request.form.get('allowsPets')),
            allowsSmoking=float(request.form.get('allowsSmoking')),
            allowsInfants=float(request.form.get('allowsInfants')),
            review_count=float(request.form.get('review_count')),
            amenities=float(request.form.get('amenities')),
            regular_bed=float(request.form.get('regular_bed')),
            relaxing_bed=float(request.form.get('relaxing_bed')),
            kids_bed=float(request.form.get('kids_bed')),
            numberofbathroom=float(request.form.get('numberofbathroom')),
            bathroomType=str(request.form.get('bathroomType'))
        )
        
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)

