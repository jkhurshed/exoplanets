# test_saved_model.py
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
import os
from datetime import datetime

def test_saved_model(pipeline_file='exominer_pipeline.pkl', model_file='exominer_model.keras'):
    """
    Test the saved ExoMiner model to verify it works correctly
    """
    
    print("=" * 80)
    print("TESTING SAVED EXOMINER MODEL")
    print("=" * 80)
    
    try:
        # Check if files exist
        if not os.path.exists(pipeline_file):
            print(f"❌ Pipeline file '{pipeline_file}' not found!")
            # Try to find any pipeline file
            import glob
            pipeline_files = glob.glob("*pipeline.pkl")
            if pipeline_files:
                pipeline_file = pipeline_files[0]
                print(f"📁 Found pipeline file: {pipeline_file}")
            else:
                return None
        
        if not os.path.exists(model_file):
            print(f"❌ Model file '{model_file}' not found!")
            # Try to find any model file
            import glob
            model_files = glob.glob("*model.keras")
            if model_files:
                model_file = model_files[0]
                print(f"📁 Found model file: {model_file}")
            else:
                model_file = None
                print("⚠️  Will try to load model from pipeline file")
        
        # Load the pipeline
        print("📥 Loading pipeline...")
        pipeline_data = joblib.load(pipeline_file)
        print("✅ Pipeline loaded successfully!")
        
        # Load the model
        if model_file and os.path.exists(model_file):
            print("📥 Loading Keras model...")
            model = keras.models.load_model(model_file)
            print("✅ Keras model loaded successfully!")
        else:
            print("📥 Loading model from pipeline...")
            model = pipeline_data['model']
            print("✅ Model loaded from pipeline!")
        
        # Get other components
        scaler = pipeline_data['scaler']
        feature_columns = pipeline_data['feature_columns']
        disposition_col = pipeline_data['disposition_column']
        accuracy = pipeline_data['accuracy']
        
        print("\n📊 MODEL INFORMATION:")
        print(f"   • Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
        print(f"   • Precision: {pipeline_data.get('precision', 'N/A')}")
        print(f"   • Recall:    {pipeline_data.get('recall', 'N/A')}")
        print(f"   • Features:  {len(feature_columns)}")
        print(f"   • Saved:     {pipeline_data.get('timestamp', 'Unknown')}")
        
        print(f"\n🎯 FEATURE LIST ({len(feature_columns)} features):")
        for i, feature in enumerate(feature_columns, 1):
            print(f"   {i:2d}. {feature}")
        
        # Test with some dummy data
        print("\n" + "="*80)
        print("MAKING TEST PREDICTIONS")
        print("="*80)
        
        # Create test data with the correct feature structure
        test_samples = []
        
        # Sample 1: Good candidate (high probability)
        sample1 = {
            'koi_period': 10.5,
            'koi_depth': 250.0,
            'koi_duration': 5.2,
            'koi_impact': 0.3,
            'koi_model_snr': 15.0,
            'koi_prad': 1.5,
            'koi_teq': 280.0,
            'koi_steff': 5800.0,
            'koi_slogg': 4.4,
            'koi_srad': 1.0,
            'koi_kepmag': 12.5,
            'koi_fpflag_nt': 0,
            'koi_fpflag_ss': 0,
            'koi_fpflag_co': 0,
            'koi_fpflag_ec': 0
        }
        
        # Sample 2: Likely false positive (low probability)
        sample2 = {
            'koi_period': 1.2,
            'koi_depth': 5000.0,
            'koi_duration': 1.5,
            'koi_impact': 0.9,
            'koi_model_snr': 5.0,
            'koi_prad': 15.0,
            'koi_teq': 1200.0,
            'koi_steff': 4500.0,
            'koi_slogg': 3.5,
            'koi_srad': 0.8,
            'koi_kepmag': 14.0,
            'koi_fpflag_nt': 1,
            'koi_fpflag_ss': 0,
            'koi_fpflag_co': 0,
            'koi_fpflag_ec': 0
        }
        
        # Only include features that exist in our feature_columns
        test_data = []
        for sample in [sample1, sample2]:
            filtered_sample = {k: v for k, v in sample.items() if k in feature_columns}
            # Fill missing features with median-like values
            for feature in feature_columns:
                if feature not in filtered_sample:
                    if 'fpflag' in feature:
                        filtered_sample[feature] = 0
                    elif 'period' in feature:
                        filtered_sample[feature] = 10.0
                    elif 'depth' in feature:
                        filtered_sample[feature] = 200.0
                    else:
                        filtered_sample[feature] = 1.0
            test_data.append(filtered_sample)
        
        test_df = pd.DataFrame(test_data)
        
        print("🧪 Test Data Created:")
        print(test_df[feature_columns[:6]].to_string())  # Show first 6 features
        
        # Prepare features
        X_test = test_df[feature_columns].copy()
        X_test = X_test.apply(pd.to_numeric, errors='coerce')
        X_test = X_test.fillna(0)  # Fill any remaining NaNs
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        print("\n📈 Making predictions...")
        probabilities = model.predict(X_test_scaled, verbose=0).flatten()
        
        print("\n🎯 PREDICTION RESULTS:")
        print("-" * 50)
        for i, prob in enumerate(probabilities, 1):
            prediction = "PLANET" if prob > 0.5 else "FALSE POSITIVE"
            confidence = "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.6 else "LOW"
            print(f"Sample {i}:")
            print(f"  • Probability: {prob:.4f} ({prob:.2%})")
            print(f"  • Prediction:  {prediction}")
            print(f"  • Confidence:  {confidence}")
            print()
        
        # Test model summary
        print("\n🔍 MODEL ARCHITECTURE:")
        model.summary()
        
        # Test with actual data file if available
        print("\n" + "="*80)
        print("TESTING WITH ACTUAL DATA FILE")
        print("="*80)
        
        if os.path.exists('koi.csv'):
            print("📁 Found koi.csv - testing with real data...")
            try:
                real_data = pd.read_csv('koi.csv', comment='#', skipinitialspace=True, on_bad_lines='skip')
                
                # Take first 5 CANDIDATE rows for testing
                candidate_mask = real_data[disposition_col] == 'CANDIDATE'
                if candidate_mask.any():
                    test_candidates = real_data[candidate_mask].head(3)
                    
                    # Prepare real data
                    X_real = test_candidates[feature_columns].copy()
                    X_real = X_real.apply(pd.to_numeric, errors='coerce')
                    X_real = X_real.fillna(X_real.median())
                    
                    # Scale features
                    X_real_scaled = scaler.transform(X_real)
                    
                    # Make predictions
                    real_probs = model.predict(X_real_scaled, verbose=0).flatten()
                    
                    print("\n🎯 REAL CANDIDATE PREDICTIONS:")
                    print("-" * 50)
                    for i, (idx, row) in enumerate(test_candidates.iterrows()):
                        prob = real_probs[i]
                        prediction = "PLANET" if prob > 0.5 else "FALSE POSITIVE"
                        confidence = "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.6 else "LOW"
                        
                        kepoi_name = row.get('kepoi_name', f'Candidate_{i+1}')
                        print(f"{kepoi_name}:")
                        print(f"  • Probability: {prob:.4f} ({prob:.2%})")
                        print(f"  • Prediction:  {prediction}")
                        print(f"  • Confidence:  {confidence}")
                        if 'koi_period' in row:
                            print(f"  • Period:      {row['koi_period']:.2f} days")
                        if 'koi_depth' in row:
                            print(f"  • Depth:       {row['koi_depth']:.2f} ppm")
                        print()
                
            except Exception as e:
                print(f"⚠️  Could not test with real data: {e}")
        
        print("✅ MODEL TEST COMPLETED SUCCESSFULLY!")
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'pipeline_data': pipeline_data,
            'test_predictions': probabilities
        }
        
    except Exception as e:
        print(f"❌ ERROR testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_predict_single(sample_data, pipeline_file='exominer_pipeline.pkl'):
    """
    Quick prediction for a single sample
    """
    try:
        pipeline_data = joblib.load(pipeline_file)
        model = pipeline_data['model']
        scaler = pipeline_data['scaler']
        feature_columns = pipeline_data['feature_columns']
        
        # Prepare the sample
        sample_df = pd.DataFrame([sample_data])
        X_sample = sample_df[feature_columns].copy()
        X_sample = X_sample.apply(pd.to_numeric, errors='coerce')
        X_sample = X_sample.fillna(0)
        
        # Scale and predict
        X_scaled = scaler.transform(X_sample)
        probability = model.predict(X_scaled, verbose=0)[0][0]
        
        return probability
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

if __name__ == "__main__":
    # Run the test
    result = test_saved_model()
    
    if result:
        print("\n🎉 SAVED MODEL WORKS CORRECTLY!")
        print("\nYou can now use this model for predictions.")

    else:
        print("\n❌ MODEL TEST FAILED!")
        print("\nMake sure you have run the training script first to create the model files.")