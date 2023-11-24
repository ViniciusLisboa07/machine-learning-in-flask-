# app/routes.py
from flask import render_template, request, Flask, jsonify
from markupsafe import Markup 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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

url = 'app/data/titanic.csv'
df = pd.read_csv(url)

columns_to_drop = ['Name', 'PassengerId', 'Cabin', 'Embarked', 'SibSp', 'Parch', 'Ticket', 'Fare']

for column in columns_to_drop:
  df = df.drop(column, axis=1)

for column in ['Age', 'Sex', 'Pclass']:
  df = df[df[column].notna()]

sex_int = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(sex_int)

X = df.drop('Survived', axis=1)
y = df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Função para gerar a matriz de confusão como uma imagem base64
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.colorbar(im)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_classifier = None
    accuracy = None
    matriz = None
    f1_score_response = None
    
    if request.method == 'POST':
        # Capturar dados do formulário
        classifier = request.form.get('classifier')
        selected_classifier = classifier
        
        if classifier == 'random_forest':
            n_estimators = int(request.form.get('n_estimators'))
            criterion = request.form.get('criterion')
            
            rf_classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
            rf_classifier.fit(X_train, y_train)
            y_pred_rf = rf_classifier.predict(X_test)
            cm_rf = confusion_matrix(y_test, y_pred_rf).tolist()
            
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            f1_score_rf = f1_score(y_test, y_pred_rf, average='macro')
            
            accuracy = accuracy_rf
            matriz = plot_confusion_matrix(cm_rf)
            f1_score_response = f1_score_rf
            
        elif classifier == 'svm':
            svm_c = float(request.form.get('svm_c'))
            svm_kernel = request.form.get('svm_kernel')
            
            svm_classifier = SVC(C=svm_c, kernel=svm_kernel)
            svm_classifier.fit(X_train, y_train)
            y_pred_svm = svm_classifier.predict(X_test)
            cm_svm = confusion_matrix(y_test, y_pred_svm).tolist()
            
            accuracy_svm = accuracy_score(y_test, y_pred_svm)
            f1_score_svm = f1_score(y_test, y_pred_svm, average='macro')
            
            accuracy = accuracy_svm
            matriz = plot_confusion_matrix(cm_svm)
            f1_score_response = f1_score_svm
            
        elif classifier == 'knn':
            knn_classifier = KNeighborsClassifier()
            knn_classifier.fit(X_train, y_train)
            y_pred_knn = knn_classifier.predict(X_test)
            cm_knn = confusion_matrix(y_test, y_pred_knn).tolist()
            
            accuracy_knn = accuracy_score(y_test, y_pred_knn)
            f1_score_knn = f1_score(y_test, y_pred_knn, average='macro')
            
            accuracy = accuracy_knn
            matriz = plot_confusion_matrix(cm_knn)
            f1_score_response = f1_score_knn
            
        elif classifier == 'mlp':
            mlp_classifier = MLPClassifier()
            mlp_classifier.fit(X_train, y_train)
            y_pred_mlp = mlp_classifier.predict(X_test)
            cm_mlp = confusion_matrix(y_test, y_pred_mlp).tolist()
            
            accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
            f1_score_mlp = f1_score(y_test, y_pred_mlp, average='macro')
            
            accuracy = accuracy_mlp
            matriz = plot_confusion_matrix(cm_mlp)
            f1_score_response = f1_score_mlp
        
        
        print("Selected Classifier:", selected_classifier)

    return render_template('index.html', selected_classifier=selected_classifier, accuracy=accuracy, matriz=matriz, f1_score_response=f1_score_response)

if __name__ == '__main__':
    app.run(debug=True)
