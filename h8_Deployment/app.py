from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask.globals import request


app = Flask(__name__, template_folder="templates")

model = pickle.load(open("model/model_clf.pkl", "rb"))

@app.route("/")
def main():
    return (Flask.render_template("main.html"))

@app.route("/predict", methods =["POST"])
def predict():
    #features_values = request.values.get('Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls',
    #   'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases',
    #   'Product_importance', 'Gender', 'Discount_offered', 'Weight_in_gms')
    #float_features = [float(x) for x in request.form.values('Cost_of_the_Product','Discount_offered','Weight_in_gms')]
    #int_features = [int(x) for x in request.form.values('Customer_care_calls','Customer_rating','Prior_purchases')]
    #str_features = [str(x) for x in request.form.values('Warehouse_block','Mode_of_Shipment','Product_importance','Gender')]
    features1 = request.form.get('Warehouse_block')
    features2 = request.form.get('Mode_of_Shipment')
    features3 = int(request.form.get('Customer_care_calls'))
    features4 = int(request.form.get('Customer_rating'))
    features5 = float(request.form.get('Cost_of_the_Product'))
    features6 = int(request.form.get('Prior_purchases'))
    features7 = request.form.get('Product_importance')
    features8 = request.form.get('Gender')
    features9 = float(request.form.get('Discount_offered'))
    features10 = float(request.form.get('Weight_in_gms'))
    features = [[features1,features2,features3,features4,features5,features6,features7,features8,features9,features10]]
    prediction = model.predict(features)

    output = {0: 'On time', 1: 'Delayed'}

    return Flask.render_template("main.html", prediction_text = "The shipment is {}".format(output[prediction[0]]))


if __name__ == "__main__":
    app.run(debug=True)    