# ml_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class MLModel:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.X = self.df.drop('target_column', axis=1)  # Substitua 'target_column' pelo nome da coluna de destino
        self.y = self.df['target_column']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.model = RandomForestClassifier()

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy
