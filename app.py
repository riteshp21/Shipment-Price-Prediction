from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from pipeline.predict_pipeline import CustomData, PredictPipeline


import logging
import sys
import os
import pandas as pd
from ShipmentPricePrediction.exception import ShipmentPriceException
import pickle
# from ShipmentPricePrediction.utils import load_object
# from ShipmentPricePrediction.logger import logging
import os
import dill

application = Flask(__name__)
app = application

# Route for a home page


"""@app.route('/')
def index():
    return render_template('index.html')"""


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # 'Item_Description'  , 'Vendor','Dosage', 'Dosage_Form',

        data = CustomData(
            Unit_of_Measure_Per_Pack=int(request.form.get('Unit_of_Measure_Per_Pack')),
            Line_Item_Quantity=int(request.form.get('Line_Item_Quantity')),
            Line_Item_Value=float(request.form.get('Line_Item_Value')),
            Pack_Price=float(request.form.get('Pack_Price')),
            Unit_Price=float(request.form.get('Unit_Price')),
            Weight_Kilograms=int(request.form.get('Weight_Kilograms')),
            Line_Item_Insurance_USD=float(request.form.get('Line_Item_Insurance_USD')),
            Country=request.form.get('Country'),
            Managed_By = request.form.get('Managed_By'),
            Fulfill_Via = request.form.get('Fulfill_Via'),
            Shipment_Mode = request.form.get('Shipment_Mode'),
            Product_Group = request.form.get('Product_Group'),
            Sub_Classification = request.form.get('Sub_Classification'),
            Brand = request.form.get('Brand'),
            First_Line_Designation = request.form.get('First_Line_Designation'))

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict_value(pred_df)
        print(results)
        print("after Prediction")
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    # app.run(host='localhost', port=5000)
    app.run(host="0.0.0.0",debug=True)

