from flask import Flask, request, render_template
import pickle

# Load the trained K-Means model and silhouette score
with open('model/kmeans_model.pkl', 'rb') as file:
    data = pickle.load(file)
    kmeans_model = data['model']
    silhouette = round(data['score'], 3)

# Initialize Flask app
application = Flask(__name__)

# Route for home page
@application.route('/')
def home():
    return render_template('home.html')

# Route for prediction
@application.route('/predict', methods=['POST'])
def predict():
    # Get form data
    gender = request.form['gender']
    age = float(request.form['age'])
    income = float(request.form['income'])
    score = float(request.form['score'])

    # Predict cluster using model trained on [Age, Income, Score]
    cluster = kmeans_model.predict([[age, income, score]])[0]

    # Render result page with all inputs and prediction
    return render_template('result.html',
                           gender=gender,
                           age=age,
                           income=income,
                           score=score,
                           cluster=cluster,
                           silhouette=silhouette)

# Run the app
if __name__ == '__main__':
    application.run()