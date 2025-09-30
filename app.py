# app.py
from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import joblib
import os
from model_training import ExoplanetModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'exoplanet_secret_key'

# Initialize model
model_handler = ExoplanetModel()
model_loaded = model_handler.load_model()

@app.route('/')
def index():
    """Main page with model information and quick prediction form"""
    model_info = {
        'loaded': model_loaded,
        'accuracy': session.get('model_accuracy', 'Not trained yet'),
        'feature_count': len(model_handler.feature_columns) if model_loaded else 0
    }
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page with form for manual data entry"""
    if request.method == 'POST':
        try:
            # Get form data
            features = []
            for col in model_handler.feature_columns:
                value = request.form.get(col, 0.0)
                features.append(float(value))
            
            # Make prediction
            prediction, probabilities = model_handler.predict(features)
            
            # Map prediction to class name
            class_names = ['False Positive', 'Planetary Candidate', 'Confirmed Exoplanet']
            prediction_name = class_names[prediction]
            
            # Create probability dictionary
            prob_dict = {
                class_names[i]: f"{prob*100:.2f}%" 
                for i, prob in enumerate(probabilities)
            }
            
            return render_template('results.html', 
                                 prediction=prediction_name,
                                 probabilities=prob_dict,
                                 features=dict(zip(model_handler.feature_columns, features)))
            
        except Exception as e:
            return render_template('predict.html', 
                                 error=f"Error making prediction: {str(e)}")
    
    return render_template('predict.html', feature_columns=model_handler.feature_columns)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions from file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read the uploaded file
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        else:
            return jsonify({'error': 'Only CSV files are supported'})
        
        # Ensure required columns are present
        missing_cols = set(model_handler.feature_columns) - set(data.columns)
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'})
        
        # Make predictions
        predictions = []
        for _, row in data.iterrows():
            features = [row[col] for col in model_handler.feature_columns]
            prediction, probabilities = model_handler.predict(features)
            class_names = ['False Positive', 'Planetary Candidate', 'Confirmed Exoplanet']
            
            predictions.append({
                'prediction': class_names[prediction],
                'confidence': f"{max(probabilities)*100:.2f}%",
                'features': row.to_dict()
            })
        
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with optional hyperparameters"""
    try:
        # Get hyperparameters from request
        n_estimators = int(request.form.get('n_estimators', 100))
        test_size = float(request.form.get('test_size', 0.2))
        
        # Retrain model
        accuracy, _, _ = model_handler.train_model()
        session['model_accuracy'] = f"{accuracy:.4f}"
        
        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'message': f'Model retrained successfully with accuracy: {accuracy:.4f}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_stats')
def model_stats():
    """Return model statistics"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'})
    
    stats = {
        'feature_columns': model_handler.feature_columns,
        'accuracy': session.get('model_accuracy', 'Not available'),
        'model_type': type(model_handler.model).__name__
    }
    
    return jsonify(stats)

@app.route('/feature_importance')
def feature_importance():
    """Generate feature importance plot"""
    if not model_loaded or not hasattr(model_handler.model, 'feature_importances_'):
        return jsonify({'error': 'Feature importance not available'})
    
    try:
        # Create feature importance plot
        importances = model_handler.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [model_handler.feature_columns[i] for i in indices], 
                  rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 for embedding in HTML
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({'plot': f"data:image/png;base64,{plot_url}"})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)