# app/routes.py
from flask import render_template, request, Flask
from .ml_model import MLModel

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        csv_path = 'data/religion-survey-results.csv'  # Substitua pelo caminho real do seu arquivo CSV
        ml_model = MLModel(csv_path)
        ml_model.train_model()
        accuracy = ml_model.test_model()
        
        return render_template('index.html', accuracy=accuracy)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
