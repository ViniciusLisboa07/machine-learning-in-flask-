# app/routes.py
from flask import render_template, request, Flask
from markupsafe import Markup 

from .ml_model import MLModel
from . import app


def render_classifier_parameters(selected_classifier):
    if selected_classifier == 'random_forest':
        return Markup(render_template('random_forest_parameters.html'))
    elif selected_classifier == 'svm':
        return Markup(render_template('svm_parameters.html'))
    # Adicione mais blocos elif para outros classificadores
    else:
        return Markup("")

app.jinja_env.globals.update(render_classifier_parameters=render_classifier_parameters)

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_classifier = None
    accuracy = None
    
    if request.method == 'POST':
        # Capturar dados do formulário
        classifier = request.form.get('classifier')
        parameter1 = request.form.get('parameter1')
        parameter2 = request.form.get('parameter2')
        
        selected_classifier = classifier
        
        if classifier == 'random_forest':
            accuracy = 0.9
        elif classifier == 'svm':
            accuracy = 0.8
        elif classifier == 'knn':
            accuracy = 0.7
        elif classifier == 'mlp':
            accuracy = 0.6
        else:
            accuracy = 0.0
        
        
        print("Selected Classifier:", selected_classifier)

        
        # Adicione mais variáveis conforme necessário

        # Use os dados do formulário para treinar o modelo
        # Exemplo básico:
        # X, y = # Carregue seus dados de treinamento e rótulos aqui
        # ml_model = MLModel(X, y)
        # ml_model.train_model()

        # Realize testes com os dados de teste (você pode precisar de dados de teste no formulário)
        # accuracy = ml_model.test_model()

    return render_template('index.html', selected_classifier=selected_classifier, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
