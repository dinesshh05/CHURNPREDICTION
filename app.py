import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('churn_prediction_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data as dict
        input_data = request.form.to_dict()

        # Convert numeric fields from string to appropriate data types
        input_data['SeniorCitizen'] = int(input_data['SeniorCitizen'])
        input_data['tenure'] = int(input_data['tenure'])
        input_data['MonthlyCharges'] = float(input_data['MonthlyCharges'])
        input_data['TotalCharges'] = float(input_data['TotalCharges'])

        # Create DataFrame
        df = pd.DataFrame([input_data])

        # Create the ordinal column (needed by the model)
        ordinal_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df['Contract_ord'] = df['Contract'].map(ordinal_map)

        # Predict using the full pipeline (preprocessing included)
        prediction = model.predict(df)[0]
        result = "Customer is likely to churn" if prediction == 1 else "Customer will stay"

        # Return prediction to frontend
        return render_template('home.html', prediction_text=result)

    except Exception as e:
        return f"Error: {e}"
 
if __name__ == "__main__":
    app.run(debug=True)
