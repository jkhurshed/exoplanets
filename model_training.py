# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class ExoplanetModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def load_and_preprocess_data(self, kepler_url, tess_url):
        """Load and combine datasets from Kepler and TESS missions"""
        try:
            # Load Kepler data
            kepler_data = pd.read_csv(kepler_url)
            kepler_data['mission'] = 'kepler'
            
            # Load TESS data
            tess_data = pd.read_csv(tess_url)
            tess_data['mission'] = 'tess'
            
            # Combine datasets
            combined_data = pd.concat([kepler_data, tess_data], ignore_index=True)
            return combined_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample data for demonstration
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for demonstration purposes"""
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'koi_period': np.random.exponential(100, n_samples),
            'koi_duration': np.random.normal(5, 2, n_samples),
            'koi_depth': np.random.exponential(500, n_samples),
            'koi_ror': np.random.exponential(0.02, n_samples),
            'koi_steff': np.random.normal(5500, 500, n_samples),
            'koi_slogg': np.random.normal(4.5, 0.5, n_samples),
            'koi_srad': np.random.normal(1.0, 0.3, n_samples),
            'koi_prad': np.random.exponential(2, n_samples),
            'koi_teq': np.random.normal(500, 200, n_samples),
            'koi_insol': np.random.exponential(1000, n_samples),
            'koi_model_snr': np.random.exponential(10, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (0: false positive, 1: candidate, 2: confirmed)
        conditions = [
            (df['koi_model_snr'] < 5) | (df['koi_depth'] < 100),
            (df['koi_model_snr'] >= 5) & (df['koi_model_snr'] < 15),
            (df['koi_model_snr'] >= 15)
        ]
        choices = [0, 1, 2]
        df['koi_disposition'] = np.select(conditions, choices, default=1)
        
        df['mission'] = np.random.choice(['kepler', 'tess'], n_samples)
        
        return df
    
    def preprocess_features(self, data):
        """Preprocess the features for training"""
        # Select relevant features
        feature_columns = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_ror',
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_prad',
            'koi_teq', 'koi_insol', 'koi_model_snr'
        ]
        
        # Handle missing values
        X = data[feature_columns].copy()
        X = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_columns
    
    def train_model(self, data_urls=None):
        """Train the exoplanet classification model"""
        if data_urls is None:
            data_urls = {
                'kepler': 'https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/koiTable.csv',
                'tess': 'https://exoplanetarchive.ipac.caltech.edu/data/TESS/koidata.csv'
            }
        
        # Load data
        print("Loading data...")
        data = self.load_and_preprocess_data(
            data_urls['kepler'], 
            data_urls['tess']
        )
        
        # Preprocess features
        print("Preprocessing features...")
        X, self.feature_columns = self.preprocess_features(data)
        y = data['koi_disposition']
        
        # Encode labels if they're strings
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models and select the best one
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_model_name = ""
        
        print("Training models...")
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"{name} accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
        
        self.model = best_model
        print(f"Best model: {best_model_name} with accuracy: {best_score:.4f}")
        
        # Save model and preprocessing objects
        self.save_model()
        
        return best_score, X_test, y_test
    
    def predict(self, features):
        """Make predictions on new data"""
        if self.model is None:
            self.load_model()
        
        # Preprocess features
        features_imputed = self.imputer.transform([features])
        features_scaled = self.scaler.transform(features_imputed)
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        joblib.dump(self.model, 'models/exoplanet_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.imputer, 'models/imputer.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.feature_columns, 'models/feature_columns.pkl')
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            self.model = joblib.load('models/exoplanet_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.imputer = joblib.load('models/imputer.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.feature_columns = joblib.load('models/feature_columns.pkl')
        except FileNotFoundError:
            print("Model not found. Please train the model first.")
            return False
        return True

# Train the model if this script is run directly
if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    
    model = ExoplanetModel()
    accuracy, X_test, y_test = model.train_model()
    print(f"Model training completed with accuracy: {accuracy:.4f}")