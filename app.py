from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
with open('random_forest_reg.pkl', 'rb') as f:
    regmodel = pickle.load(f)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template('shipp.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data and preprocess if needed
        form_data = {
            'unit_of_measure_(per_pack)': float(request.form['unit_of_measure_(per_pack)']),
            'line_item_quantity': float(request.form['line_item_quantity']),
            'pack_price': float(request.form['pack_price']),
            'unit_price': float(request.form['unit_price']),
            'freight_cost_(usd)': float(request.form['freight_cost_(usd)']),
            'line_item_insurance_(usd)': float(request.form['line_item_insurance_(usd)']),
            'first_line_designation': int(request.form['first_line_designation']),
            'shipment_mode': int(request.form['shipment_mode']),
            'manufacturing_site': int(request.form['manufacturing_site'])
        }


        # Apply label encoding to categorical fields
        # Note: Make sure the label_encoders dictionary contains the appropriate encoders
        for column in ['first_line_designation', 'shipment_mode', 'manufacturing_site']:
            form_data[column] = label_encoders[column].transform([str(form_data[column])])[0]

        # Prepare data for prediction
        data = list(form_data.values())
        final_input = np.array(data).reshape(1, -1)

        # Make prediction
        output = regmodel.predict(final_input)[0]

        # Render the template with the prediction
        return render_template('shipp.html', prediction_text="The Shipment price prediction is ${:.2f}".format(output))
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
