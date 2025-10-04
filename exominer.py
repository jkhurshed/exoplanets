import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from model_saving import save_exominer_pipeline, save_predictions

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=== ExoMiner-Inspired Exoplanet Hunter ===")
print("Loading and preprocessing data...")

# Step 1: Load and preprocess the data
def load_and_preprocess_data():
    """Load and preprocess the KOI data"""
    try:
        df = pd.read_csv('koi.csv', comment='#', skipinitialspace=True, on_bad_lines='skip')
        print(f"✓ Successfully loaded DataFrame with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        try:
            df = pd.read_csv('koi.csv', comment='#', engine='python', error_bad_lines=False)
            print(f"✓ Alternative load successful. Shape: {df.shape}")
            return df
        except Exception as e2:
            print(f"✗ All loading methods failed: {e2}")
            return None

# Load the data
df = load_and_preprocess_data()
if df is None:
    exit()

# Find disposition column
disposition_cols = [col for col in df.columns if 'disposition' in col.lower()]
if not disposition_cols:
    print("✗ No disposition column found!")
    exit()

disp_col = disposition_cols[0]
print(f"✓ Using disposition column: {disp_col}")
print(f"Disposition value counts:\n{df[disp_col].value_counts()}")

# Step 2: Feature engineering and selection
print("\n=== Feature Engineering ===")

# Create engineered features
df['koi_period_depth_ratio'] = df['koi_period'] / (df['koi_depth'] + 1e-6)
df['koi_duration_period_ratio'] = df['koi_duration'] / (df['koi_period'] + 1e-6)
df['koi_radius_flux_ratio'] = df['koi_prad'] / (df['koi_insol'] + 1e-6)

# Handle uncertainty features
df['koi_period_err_rel'] = (df['koi_period_err1'] - df['koi_period_err2']).abs() / (df['koi_period'] + 1e-6)
df['koi_depth_err_rel'] = (df['koi_depth_err1'] - df['koi_depth_err2']).abs() / (df['koi_depth'] + 1e-6)

# Define feature groups
transit_features = [
    'koi_period', 'koi_depth', 'koi_duration', 'koi_impact', 
    'koi_model_snr', 'koi_period_depth_ratio', 'koi_duration_period_ratio'
]

uncertainty_features = [
    'koi_period_err_rel', 'koi_depth_err_rel'
]

stellar_features = [
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag'
]

planetary_features = [
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_radius_flux_ratio'
]

false_positive_flags = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]

# Combine all features (only use existing columns)
all_possible_features = (transit_features + uncertainty_features + 
                        stellar_features + planetary_features + false_positive_flags)

feature_columns = [col for col in all_possible_features if col in df.columns]
print(f"✓ Using {len(feature_columns)} features for training")

# Step 3: Prepare training data
print("\n=== Preparing Training Data ===")

# Filter for training data (CANDIDATE and FALSE POSITIVE)
valid_dispositions = ['CANDIDATE', 'FALSE POSITIVE']
train_mask = df[disp_col].isin(valid_dispositions)
df_train = df[train_mask].copy()

if len(df_train) == 0:
    print("✗ No training data found!")
    exit()

print(f"✓ Training samples: {len(df_train)}")
print(f"  - CANDIDATE: {sum(df_train[disp_col] == 'CANDIDATE')}")
print(f"  - FALSE POSITIVE: {sum(df_train[disp_col] == 'FALSE POSITIVE')}")

# Create target variable
df_train['is_planet'] = df_train[disp_col].map({
    'CANDIDATE': 1,
    'FALSE POSITIVE': 0
})

# Prepare features and target
X = df_train[feature_columns]
y = df_train['is_planet']

# Handle missing values
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())

# Remove any rows with remaining NaN values
valid_mask = ~X.isnull().any(axis=1)
X = X[valid_mask]
y = y[valid_mask]

print(f"✓ Final training set: {X.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set: {X_train_scaled.shape}")
print(f"✓ Test set: {X_test_scaled.shape}")

# Step 4: Build and train the model
print("\n=== Building ExoMiner Model ===")

def create_exominer_model(input_dim):
    """Create the neural network model"""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Create the model
model = create_exominer_model(X_train_scaled.shape[1])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("✓ Model architecture created")
model.summary()

# Train the model
print("\n=== Training Model ===")

callbacks = [
    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
]

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
print("\n=== Model Evaluation ===")
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    X_test_scaled, y_test, verbose=0
)

print(f"✓ Test Accuracy: {test_accuracy:.4f}")
print(f"✓ Test Precision: {test_precision:.4f}")
print(f"✓ Test Recall: {test_recall:.4f}")

# Step 5: Make Predictions on New Candidates
print("\n" + "="*80)
print("MAKING PREDICTIONS ON CANDIDATES")
print("="*80)

def predict_candidates(model, scaler, original_df, feature_columns, disposition_column):
    """Make predictions on candidate exoplanets"""
    
    candidates_mask = original_df[disposition_column] == 'CANDIDATE'
    
    if not candidates_mask.any():
        print("✗ No CANDIDATE dispositions found for prediction")
        return None
    
    candidates_df = original_df[candidates_mask].copy()
    print(f"✓ Found {len(candidates_df)} candidate systems for prediction")
    
    # Prepare candidate features
    X_candidates = candidates_df[feature_columns].copy()
    X_candidates = X_candidates.apply(pd.to_numeric, errors='coerce')
    X_candidates = X_candidates.fillna(X_candidates.median())
    
    # Scale features
    X_candidates_scaled = scaler.transform(X_candidates)
    
    # Get probability scores
    candidate_probabilities = model.predict(X_candidates_scaled, verbose=0).flatten()
    candidates_df['planet_probability'] = candidate_probabilities
    candidates_df['predicted_class'] = (candidate_probabilities > 0.5).astype(int)
    
    # Add confidence levels
    candidates_df['confidence'] = candidates_df['planet_probability'].apply(
        lambda x: 'HIGH' if x > 0.8 else 'MEDIUM' if x > 0.6 else 'LOW'
    )
    
    return candidates_df

# Make predictions
candidates_with_predictions = predict_candidates(model, scaler, df, feature_columns, disp_col)

if candidates_with_predictions is not None:
    # Display most promising candidates
    print("\nTOP 20 MOST PROMISING EXOPLANET CANDIDATES:")
    print("-" * 80)
    
    promising_candidates = candidates_with_predictions.nlargest(20, 'planet_probability')[
        ['kepoi_name', 'kepler_name', 'planet_probability', 'confidence', 
         'koi_period', 'koi_depth', 'koi_model_snr', 'koi_prad', 'koi_teq']
    ].copy()
    
    # Format for display
    promising_candidates['planet_probability_pct'] = (promising_candidates['planet_probability'] * 100).round(2)
    
    print(promising_candidates[['kepoi_name', 'kepler_name', 'planet_probability_pct', 'confidence']].to_string(index=False))
    
    # Save detailed predictions
    output_columns = ['kepoi_name', 'kepler_name', 'planet_probability', 'confidence', 
                     'predicted_class', 'koi_period', 'koi_depth', 'koi_duration', 
                     'koi_model_snr', 'koi_prad', 'koi_teq', 'koi_steff']
    
    available_output_columns = [col for col in output_columns if col in candidates_with_predictions.columns]
    candidates_with_predictions[available_output_columns].to_csv('exoplanet_candidate_predictions.csv', index=False)
    print(f"\n✓ Detailed predictions saved to 'exoplanet_candidate_predictions.csv'")

# Step 6: Fixed Feature Importance Analysis (No Drop-Column Method)
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

def analyze_feature_importance_safe(model, feature_columns, X_train_scaled, y_train):
    """Analyze feature importance using safe methods that don't require model retraining"""
    
    print("Calculating feature importance using safe methods...")
    
    # Method 1: Neural Network Weight Analysis (Most reliable)
    weights = model.layers[0].get_weights()[0]  # Weights from first layer
    weight_importance = np.mean(np.abs(weights), axis=1)
    
    weight_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'weight_importance': weight_importance
    }).sort_values('weight_importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Neural Network Weights):")
    print(weight_importance_df.head(10).round(4))
    
    # Method 2: Correlation Analysis
    correlations = []
    for i, feature in enumerate(feature_columns):
        if hasattr(X_train_scaled, 'shape'):
            corr = np.corrcoef(X_train_scaled[:, i], y_train)[0, 1]
            correlations.append(abs(corr))
        else:
            correlations.append(0)
    
    correlation_df = pd.DataFrame({
        'feature': feature_columns,
        'correlation_with_target': correlations
    }).sort_values('correlation_with_target', ascending=False)
    
    print("\nTop 10 Most Important Features (Correlation with Target):")
    print(correlation_df.head(10).round(4))
    
    # Method 3: SHAP-like approximation using model gradients
    print("\nCalculating gradient-based importance...")
    
    # Convert to TensorFlow tensor for gradient computation
    X_train_tensor = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_train_tensor)
        predictions = model(X_train_tensor)
    
    # Compute gradients of predictions with respect to inputs
    gradients = tape.gradient(predictions, X_train_tensor)
    
    # Average absolute gradients across samples
    gradient_importance = np.mean(np.abs(gradients.numpy()), axis=0)
    
    gradient_df = pd.DataFrame({
        'feature': feature_columns,
        'gradient_importance': gradient_importance
    }).sort_values('gradient_importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Gradient-based):")
    print(gradient_df.head(10).round(4))
    
    # Combine all methods
    combined_importance = pd.merge(weight_importance_df, correlation_df, on='feature')
    combined_importance = pd.merge(combined_importance, gradient_df, on='feature')
    
    # Normalize and combine scores
    for col in ['weight_importance', 'correlation_with_target', 'gradient_importance']:
        if combined_importance[col].max() > 0:
            combined_importance[col + '_norm'] = combined_importance[col] / combined_importance[col].max()
        else:
            combined_importance[col + '_norm'] = 0
    
    combined_importance['combined_score'] = (
        combined_importance['weight_importance_norm'] + 
        combined_importance['correlation_with_target_norm'] + 
        combined_importance['gradient_importance_norm']
    )
    
    combined_importance = combined_importance.sort_values('combined_score', ascending=False)
    
    return combined_importance

# Perform feature importance analysis
importance_results = analyze_feature_importance_safe(model, feature_columns, X_train_scaled, y_train)

# Save results
importance_results.to_csv('feature_importance_analysis.csv', index=False)
print(f"\n✓ Feature importance analysis saved to 'feature_importance_analysis.csv'")

# Step 7: Enhanced Visualizations
print("\n=== Creating Enhanced Visualizations ===")

plt.figure(figsize=(16, 12))

# Plot 1: Combined Feature Importance
plt.subplot(2, 2, 1)
top_combined = importance_results.head(15)
plt.barh(range(len(top_combined)), top_combined['combined_score'], alpha=0.7, color='steelblue')
plt.yticks(range(len(top_combined)), top_combined['feature'])
plt.xlabel('Combined Importance Score')
plt.title('Top 15 Most Important Features (Combined Score)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)

# Plot 2: Training History
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.legend()
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Plot 3: Candidate Probabilities Distribution
if candidates_with_predictions is not None:
    plt.subplot(2, 2, 3)
    n, bins, patches = plt.hist(candidates_with_predictions['planet_probability'], 
                               bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Planet Probability')
    plt.ylabel('Number of Candidates')
    plt.title('Distribution of Candidate Probabilities')
    plt.grid(True, alpha=0.3)
    
    # Color code the bins by confidence
    for i, patch in enumerate(patches):
        if bins[i] > 0.8:
            patch.set_facecolor('green')
        elif bins[i] > 0.6:
            patch.set_facecolor('orange')

# Plot 4: Feature Importance Method Comparison
plt.subplot(2, 2, 4)
top_features = importance_results.head(8)['feature'].tolist()

weight_norm = importance_results.head(8)['weight_importance_norm'].values
corr_norm = importance_results.head(8)['correlation_with_target_norm'].values
gradient_norm = importance_results.head(8)['gradient_importance_norm'].values

x = np.arange(len(top_features))
width = 0.25

plt.bar(x - width, weight_norm, width, label='Weights', alpha=0.8)
plt.bar(x, corr_norm, width, label='Correlation', alpha=0.8)
plt.bar(x + width, gradient_norm, width, label='Gradient', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Normalized Importance')
plt.title('Feature Importance Method Comparison')
plt.xticks(x, top_features, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exoplanet_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 8: Generate Insights Report
print("\n" + "="*80)
print("ANALYSIS INSIGHTS AND SUMMARY")
print("="*80)

print("\n🔍 KEY FINDINGS:")
print(f"• Model achieved {test_accuracy:.1%} accuracy on test data")
print(f"• {test_precision:.1%} precision - correctly identifies true planets")
print(f"• {test_recall:.1%} recall - finds most actual planets")

if candidates_with_predictions is not None:
    high_conf = sum(candidates_with_predictions['confidence'] == 'HIGH')
    med_conf = sum(candidates_with_predictions['confidence'] == 'MEDIUM')
    low_conf = sum(candidates_with_predictions['confidence'] == 'LOW')
    
    print(f"\n📊 CANDIDATE BREAKDOWN:")
    print(f"• {len(candidates_with_predictions)} total candidates analyzed")
    print(f"• {high_conf} high-confidence planet candidates (>80% probability)")
    print(f"• {med_conf} medium-confidence candidates (60-80% probability)")
    print(f"• {low_conf} low-confidence candidates (<60% probability)")

print(f"\n🎯 TOP 5 MOST IMPORTANT FEATURES:")
top_5_features = importance_results.head(5)
for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
    print(f"  {i}. {row['feature']} (score: {row['combined_score']:.3f})")

print(f"\n💡 INTERPRETATION OF IMPORTANT FEATURES:")
top_features_list = importance_results.head(3)['feature'].tolist()
for feature in top_features_list:
    if 'fpflag' in feature:
        print(f"  • {feature}: Direct false positive indicator")
    elif 'snr' in feature.lower():
        print(f"  • {feature}: Signal quality measure")
    elif 'depth' in feature.lower():
        print(f"  • {feature}: Transit signal strength")
    elif 'period' in feature.lower():
        print(f"  • {feature}: Orbital characteristics")
    elif 'teq' in feature.lower() or 'steff' in feature.lower():
        print(f"  • {feature}: Temperature-related feature")
    else:
        print(f"  • {feature}: Important predictive feature")

print("Successfully completed exoplanet analysis and model learning!")


save_result = save_exominer_pipeline(
    model=model,
    scaler=scaler,
    feature_columns=feature_columns,
    disposition_col=disp_col,
    accuracy=test_accuracy,
    precision=test_precision,
    recall=test_recall,
    feature_importance_df=importance_results,
    history=history,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="exominer"
)

# Save candidate predictions
if candidates_with_predictions is not None:
    save_predictions(candidates_with_predictions)