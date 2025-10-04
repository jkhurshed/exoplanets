# save_exominer.py
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras
import os
from datetime import datetime

def save_exominer_pipeline(model, scaler, feature_columns, disposition_col, 
                          accuracy, precision, recall, feature_importance_df,
                          history, X_train, y_train, X_test, y_test,
                          model_name="exominer"):
    """
    Save the complete ExoMiner pipeline to files
    
    Parameters:
    - model: Trained Keras model
    - scaler: Fitted StandardScaler
    - feature_columns: List of feature names used
    - disposition_col: Name of the target column
    - accuracy: Model accuracy
    - precision: Model precision
    - recall: Model recall
    - feature_importance_df: DataFrame with feature importance
    - history: Training history
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    - model_name: Base name for saved files
    """
    
    print("=" * 80)
    print("SAVING EXOMINER PIPELINE")
    print("=" * 80)
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_name = f"{model_name}_{timestamp}"
    
    try:
        # Method 1: Save complete pipeline as PKL
        pipeline_data = {
            # Model components
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'disposition_column': disposition_col,
            
            # Performance metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'feature_importance': feature_importance_df,
            
            # Training info
            'training_history': history.history if hasattr(history, 'history') else history,
            'input_shape': model.input_shape,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            
            # Metadata
            'timestamp': timestamp,
            'creation_date': datetime.now().isoformat(),
            'class_distribution': {
                'training_planets': int(y_train.sum()),
                'training_false_positives': len(y_train) - int(y_train.sum()),
                'test_planets': int(y_test.sum()),
                'test_false_positives': len(y_test) - int(y_test.sum())
            },
            
            # Data info
            'feature_stats': {
                'num_features': len(feature_columns),
                'features': feature_columns,
                'feature_types': 'transit_parameters_stellar_properties'
            }
        }
        
        # Save pipeline
        pipeline_filename = f"{versioned_name}_pipeline.pkl"
        joblib.dump(pipeline_data, pipeline_filename)
        print(f"✓ 1. Complete pipeline saved as '{pipeline_filename}'")
        
        # Method 2: Save Keras model
        model_filename = f"{versioned_name}_model.keras"
        model.save(model_filename)
        print(f"✓ 2. Keras model saved as '{model_filename}'")
        
        # Method 3: Save model weights
        weights_filename = f"{versioned_name}_weights.h5"
        model.save_weights(weights_filename)
        print(f"✓ 3. Model weights saved as '{weights_filename}'")
        
        # Method 4: Save metadata as JSON
        metadata = {
            'model_info': {
                'name': 'ExoMiner Exoplanet Classifier',
                'version': versioned_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'input_shape': [dim for dim in model.input_shape if dim is not None],
                'output_type': 'binary_classification'
            },
            'features': {
                'count': len(feature_columns),
                'names': feature_columns,
                'disposition_column': disposition_col
            },
            'training_info': {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'planet_candidates_training': int(y_train.sum()),
                'false_positives_training': len(y_train) - int(y_train.sum()),
                'timestamp': timestamp
            },
            'top_features': feature_importance_df.head(10)[['feature', 'combined_score']].to_dict('records')
        }
        
        metadata_filename = f"{versioned_name}_metadata.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ 4. Metadata saved as '{metadata_filename}'")
        
        # Method 5: Save feature importance separately
        importance_filename = f"{versioned_name}_feature_importance.csv"
        feature_importance_df.to_csv(importance_filename, index=False)
        print(f"✓ 5. Feature importance saved as '{importance_filename}'")
        
        # Create a summary file
        summary_filename = f"{versioned_name}_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write("EXOMINER MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {versioned_name}\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})\n")
            f.write(f"Precision: {precision:.4f} ({precision:.2%})\n")
            f.write(f"Recall:    {recall:.4f} ({recall:.2%})\n\n")
            
            f.write("DATA STATISTICS:\n")
            f.write(f"Training samples: {len(X_train)}\n")
            f.write(f"Test samples:     {len(X_test)}\n")
            f.write(f"Features:         {len(feature_columns)}\n")
            f.write(f"Input shape:      {model.input_shape}\n\n")
            
            f.write("TOP 5 FEATURES:\n")
            for i, row in feature_importance_df.head().iterrows():
                f.write(f"{i+1}. {row['feature']} (score: {row['combined_score']:.3f})\n")
        
        print(f"✓ 6. Summary saved as '{summary_filename}'")
        
        # Create latest symlinks (optional)
        try:
            os.system(f"ln -sf {pipeline_filename} {model_name}_latest_pipeline.pkl")
            os.system(f"ln -sf {model_filename} {model_name}_latest_model.keras")
            print("✓ 7. Latest symlinks created")
        except:
            print("⚠  Could not create symlinks (non-Unix system)")
        
        print(f"\n💾 SAVE COMPLETE! Files created:")
        print(f"   • {pipeline_filename}")
        print(f"   • {model_filename}")
        print(f"   • {weights_filename}")
        print(f"   • {metadata_filename}")
        print(f"   • {importance_filename}")
        print(f"   • {summary_filename}")
        
        return {
            'pipeline_file': pipeline_filename,
            'model_file': model_filename,
            'weights_file': weights_filename,
            'metadata_file': metadata_filename,
            'importance_file': importance_filename,
            'summary_file': summary_filename,
            'version': versioned_name
        }
        
    except Exception as e:
        print(f"✗ Error saving pipeline: {e}")
        return None

def save_predictions(candidates_df, model_name="exominer"):
    """Save candidate predictions to file"""
    if candidates_df is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_filename = f"{model_name}_predictions_{timestamp}.csv"
        
        # Select relevant columns
        output_columns = ['kepoi_name', 'kepler_name', 'planet_probability', 'confidence', 
                         'predicted_class', 'koi_period', 'koi_depth', 'koi_duration', 
                         'koi_model_snr', 'koi_prad', 'koi_teq', 'koi_steff']
        
        available_columns = [col for col in output_columns if col in candidates_df.columns]
        candidates_df[available_columns].to_csv(predictions_filename, index=False)
        
        print(f"✓ Predictions saved as '{predictions_filename}'")
        return predictions_filename
    return None

if __name__ == "__main__":
    print("This file contains saving functions for ExoMiner pipeline.")
    print("Import these functions in your main training script.")