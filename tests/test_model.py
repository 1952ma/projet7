import unittest
import joblib
import os
import pandas as pd

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        """Configuration avant chaque test."""
        # Définir les chemins du modèle et du scaler
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(current_directory, "..", "api", "lightgbm_model_f.joblib")
        self.scaler_path = os.path.join(current_directory, "..", "api", "scaler_lgbm.joblib")
    
    def test_model_exists(self):
        """Test si le modèle a bien été sauvegardé."""
        self.assertTrue(os.path.exists(self.model_path), "Le modèle n'a pas été sauvegardé correctement.")
    
    def test_scaler_exists(self):
        """Test si le scaler a bien été sauvegardé."""
        self.assertTrue(os.path.exists(self.scaler_path), "Le scaler n'a pas été sauvegardé correctement.")
    
    def test_scaler_transformation(self):
        """Test si le scaler transforme correctement les données."""
        # Charger le scaler
        scaler = joblib.load(self.scaler_path)

        # Charger les données du fichier CSV et retirer la colonne SK_ID_CURR
        current_directory = os.path.dirname(os.path.realpath(__file__))
        X_sample = pd.read_csv(os.path.join(current_directory, "..", "api", "df_clients.csv")).drop(columns=["SK_ID_CURR"])

        # Appliquer la transformation du scaler
        X_scaled = scaler.transform(X_sample)
        
        # Vérifier que les données ont bien été transformées
        self.assertEqual(X_scaled.shape, X_sample.shape, "La transformation du scaler a échoué.")
    
    def test_model_prediction(self):
        """Test si le modèle peut prédire avec des données du fichier CSV."""
        # Charger le modèle
        model = joblib.load(self.model_path)

        # Charger les données du fichier CSV, retirer la colonne SK_ID_CURR
        current_directory = os.path.dirname(os.path.realpath(__file__))
        X_sample = pd.read_csv(os.path.join(current_directory, "..", "api", "df_clients.csv")).drop(columns=["SK_ID_CURR"])

        # Test simple de prédiction
        predictions = model.predict(X_sample)
    
        # Vérifier que le modèle renvoie le bon nombre de prédictions
        self.assertEqual(len(predictions), len(X_sample), "La prédiction du modèle a échoué.")

if __name__ == '__main__':
    unittest.main()


