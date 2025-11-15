"""
Batch Prediction Script - Predict on Unseen CSV Data
Non-interactive CLI tool for production batch predictions

Usage:
    python src/predict_unseen.py <input_csv> <output_csv>
    
Example:
    python src/predict_unseen.py data/unseen.csv data/predictions_output.csv
"""

import sys
import os
import pandas as pd
import joblib
from datetime import datetime

def print_banner():
    """Print script banner"""
    print("\n" + "="*70)
    print("🌞 Solar PV Batch Prediction Tool")
    print("="*70)

def validate_args():
    """Validate command-line arguments"""
    if len(sys.argv) != 3:
        print("❌ Error: Invalid number of arguments")
        print("\nUsage:")
        print("  python src/predict_unseen.py <input_csv> <output_csv>")
        print("\nExample:")
        print("  python src/predict_unseen.py data/unseen.csv data/predictions.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Check if input is CSV
    if not input_path.endswith('.csv'):
        print(f"❌ Error: Input file must be a CSV file: {input_path}")
        sys.exit(1)
    
    # Check if output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"❌ Error: Output directory does not exist: {output_dir}")
        print(f"   Creating directory...")
        os.makedirs(output_dir, exist_ok=True)
        print(f"   ✅ Directory created")
    
    return input_path, output_path

def load_model():
    """Load the trained model artifact"""
    model_path = 'models/pipeline_prod.joblib'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python src/train_production.py")
        sys.exit(1)
    
    print(f"\n📦 Loading model from: {model_path}")
    try:
        artifact = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Version: {artifact['version']}")
        print(f"   Features: {len(artifact['features'])}")
        return artifact
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        sys.exit(1)

def load_input_data(input_path):
    """Load and validate input CSV"""
    print(f"\n📂 Loading input data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Data loaded successfully")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        return df
    except pd.errors.EmptyDataError:
        print(f"❌ Error: Input file is empty: {input_path}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"❌ Error parsing CSV: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        sys.exit(1)

def validate_features(df, required_features):
    """Validate that all required features are present"""
    print(f"\n🔍 Validating features...")
    
    df_features = set(df.columns)
    required_features_set = set(required_features)
    
    missing = required_features_set - df_features
    extra = df_features - required_features_set
    
    if missing:
        print(f"❌ Error: Missing required features:")
        for feat in missing:
            print(f"   - {feat}")
        print(f"\nRequired features:")
        for feat in required_features:
            print(f"   - {feat}")
        sys.exit(1)
    
    if extra:
        print(f"⚠️  Warning: Extra columns in input (will be preserved in output):")
        for feat in extra:
            print(f"   - {feat}")
    
    print(f"✅ All required features present")

def make_predictions(df, artifact):
    """Make predictions on the input data"""
    print(f"\n🔮 Making predictions...")
    
    pipeline = artifact['pipeline']
    features = artifact['features']
    
    try:
        # Extract features in correct order
        X = df[features]
        
        # Check for missing values
        if X.isnull().any().any():
            print("⚠️  Warning: Input data contains missing values")
            missing_counts = X.isnull().sum()
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"   - {col}: {count} missing values")
            print("   Filling missing values with column means...")
            X = X.fillna(X.mean())
        
        # Make predictions
        predictions = pipeline.predict(X)
        
        print(f"✅ Predictions completed successfully")
        print(f"   Predictions: {len(predictions)}")
        
        # Add statistics
        print(f"\n📊 Prediction Statistics:")
        print(f"   Mean:   {predictions.mean():.2f} W")
        print(f"   Median: {pd.Series(predictions).median():.2f} W")
        print(f"   Min:    {predictions.min():.2f} W")
        print(f"   Max:    {predictions.max():.2f} W")
        print(f"   Std:    {predictions.std():.2f} W")
        
        return predictions
        
    except KeyError as e:
        print(f"❌ Error: Feature mismatch - {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        sys.exit(1)

def save_output(df, predictions, output_path, model_version):
    """Save predictions to output CSV"""
    print(f"\n💾 Saving results to: {output_path}")
    
    try:
        # Create output dataframe (copy of input)
        output_df = df.copy()
        
        # Add prediction column
        output_df['Predicted_Power_W'] = predictions.round(2)
        
        # Add metadata columns
        output_df['Model_Version'] = model_version
        output_df['Prediction_Timestamp'] = datetime.now().isoformat()
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        
        print(f"✅ Results saved successfully")
        print(f"   Output file: {output_path}")
        print(f"   Total rows: {len(output_df)}")
        print(f"   Total columns: {len(output_df.columns)}")
        
        # Show sample predictions
        print(f"\n📋 Sample Predictions (first 5 rows):")
        print(output_df[['Predicted_Power_W', 'Model_Version']].head().to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error saving output: {str(e)}")
        sys.exit(1)

def main():
    """Main execution function"""
    print_banner()
    
    # Step 1: Validate arguments
    input_path, output_path = validate_args()
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    
    # Step 2: Load model
    artifact = load_model()
    
    # Step 3: Load input data
    df = load_input_data(input_path)
    
    # Step 4: Validate features
    validate_features(df, artifact['features'])
    
    # Step 5: Make predictions
    predictions = make_predictions(df, artifact)
    
    # Step 6: Save output
    save_output(df, predictions, output_path, artifact['version'])
    
    # Success message
    print("\n" + "="*70)
    print("🎉 Batch prediction completed successfully!")
    print("="*70)
    print(f"\nOutput file ready: {output_path}")
    print("\nNext steps:")
    print("  1. Review the output CSV file")
    print("  2. Analyze prediction statistics")
    print("  3. Visualize results if needed")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)