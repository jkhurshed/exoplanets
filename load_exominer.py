# load_exominer.py
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
import json
from datetime import datetime

def load_exominer_pipeline(pipeline_file=None, model_file=None):
    """
    Load ExoMiner pipeline from saved files
    
    Parameters:
    - pipeline_file: Path to .pkl pipeline file
    - model_file: Path to .keras model file (optional)
    
    Returns:
    - Dictionary with loaded components
    """
    
    print("=" * 80)
    print("LOADING EXOMINER PIPELINE")
    print("=" * 80)
    
    try:
        if pipeline_file is None:
            # Try to find the latest pipeline file
            import glob
            pipeline_files = glob.glob("exominer_*_pipeline.pkl")
            if pipeline_files:
                pipeline_file = sorted(pipeline_files)[-1]  # Get most recent
                print(f"Found pipeline file: {pipeline_file}")
            else:
                raise FileNotFoundError("No pipeline files found")
        
        # Load pipeline data
        pipeline_data = joblib.load(pipeline_file)
        print("✓ Pipeline data loaded successfully")
        
        # Load model (either from separate file or from pipeline)
        if model_file:
            model = keras.models.load_model(model_file)
            print(f"✓ Model loaded from: {model_file}")
        else:
            model = pipeline_data['model']
            print("✓ Model loaded from pipeline")
        
        # Extract components
        scaler = pipeline_data['scaler']
        feature_columns = pipeline_data['feature_columns']
        disposition_col = pipeline_data['disposition_column']
        
        # Print summary
        print(f"\n📊 MODEL INFORMATION:")
        print(f"• Accuracy:  {pipeline_data['accuracy']:.4f}")
        print(f"• Precision: {pipeline_data['precision']:.4f}")
        print(f"• Recall:    {pipeline_data['recall']:.4f}")
        print(f"• Features:  {len(feature_columns)}")
        print(f"• Input shape: {pipeline_data['input_shape']}")
        print(f"• Created: {pipeline_data.get('timestamp', 'Unknown')}")
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'disposition_column': disposition_col,
            'pipeline_data': pipeline_data,
            'accuracy': pipeline_data['accuracy'],
            'precision': pipeline_data['precision'],
            'recall': pipeline_data['recall']
        }
        
    except Exception as e:
        print(f"✗ Error loading pipeline: {e}")
        return None

def predict_new_candidates(new_data, pipeline_components, return_probabilities=True):
    """
    Make predictions on new candidate data
    
    Parameters:
    - new_data: DataFrame with new KOI data
    - pipeline_components: Output from load_exominer_pipeline()
    - return_probabilities: Whether to return probabilities or binary predictions
    
    Returns:
    - DataFrame with predictions
    """
    
    try:
        model = pipeline_components['model']
        scaler = pipeline_components['scaler']
        feature_columns = pipeline_components['feature_columns']
        
        print(f"Making predictions on {len(new_data)} candidates...")
        
        # Prepare features
        X_new = new_data[feature_columns].copy()
        X_new = X_new.apply(pd.to_numeric, errors='coerce')
        X_new = X_new.fillna(X_new.median())
        
        # Scale features
        X_new_scaled = scaler.transform(X_new)
        
        # Make predictions
        probabilities = model.predict(X_new_scaled, verbose=0).flatten()
        
        # Create results
        results = new_data.copy()
        results['planet_probability'] = probabilities
        results['predicted_class'] = (probabilities > 0.5).astype(int)
        results['confidence'] = results['planet_probability'].apply(
            lambda x: 'HIGH' if x > 0.8 else 'MEDIUM' if x > 0.6 else 'LOW'
        )
        
        # Add model info
        results['model_accuracy'] = pipeline_components['accuracy']
        results['prediction_timestamp'] = datetime.now()
        
        print(f"✓ Predictions completed: {len(results)} candidates processed")
        print(f"  - High confidence: {sum(results['confidence'] == 'HIGH')}")
        print(f"  - Medium confidence: {sum(results['confidence'] == 'MEDIUM')}")
        print(f"  - Low confidence: {sum(results['confidence'] == 'LOW')}")
        
        return results
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return None

def get_model_info(pipeline_file=None):
    """Get basic information about saved model"""
    try:
        if pipeline_file is None:
            import glob
            pipeline_files = glob.glob("exominer_*_pipeline.pkl")
            if pipeline_files:
                pipeline_file = sorted(pipeline_files)[-1]
            else:
                return None
        
        pipeline_data = joblib.load(pipeline_file)
        
        info = {
            'accuracy': pipeline_data['accuracy'],
            'precision': pipeline_data['precision'],
            'recall': pipeline_data['recall'],
            'features': pipeline_data['feature_columns'],
            'num_features': len(pipeline_data['feature_columns']),
            'timestamp': pipeline_data.get('timestamp', 'Unknown'),
            'training_samples': pipeline_data.get('training_samples', 'Unknown')
        }
        
        return info
        
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None

if __name__ == "__main__":
    # Test loading
    components = load_exominer_pipeline()
    if components:
        print("\nModel loaded successfully!")
        
        # Show available functions
        print("\nAvailable functions:")
        print("• load_exominer_pipeline() - Load saved model")
        print("• predict_new_candidates() - Make predictions on new data")
        print("• get_model_info() - Get model metadata")