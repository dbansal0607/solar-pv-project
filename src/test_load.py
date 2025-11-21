"""
Test Model Loading and Make Sample Prediction
Verifies that the saved model artifact can be loaded and used

Usage:
    python src/test_load.py
"""

import joblib
import pandas as pd
import json
import os


def main():
    print("\n" + "=" * 70)
    print("🧪 Model Loading Test")
    print("=" * 70)

    # Check if model file exists
    model_path = "models/pipeline_prod.joblib"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python src/train_production.py")
        return

    print(f"✅ Model file found: {model_path}")

    # Load the model artifact
    print("\n📦 Loading model artifact...")
    try:
        artifact = joblib.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

    # Inspect artifact structure
    print("\n📋 Artifact Structure:")
    print(f"  Keys: {list(artifact.keys())}")
    print(f"  Model Version: {artifact['version']}")
    print(f"  Training Date: {artifact['train_date']}")
    print(f"  Training Samples: {artifact['training_samples']}")
    print(f"  Features ({len(artifact['features'])}):")
    for i, feature in enumerate(artifact["features"], 1):
        print(f"    {i}. {feature}")

    # Extract pipeline
    pipeline = artifact["pipeline"]
    features = artifact["features"]

    # Create sample input
    print("\n🧪 Testing prediction with sample data...")
    sample_data = {
        "Solar_Irradiance_kWh_m2": 0.75,
        "Temperature_C": 25.0,
        "Wind_Speed_mps": 3.5,
        "Relative_Humidity_%": 65.0,
        "Panel_Tilt_deg": 30.0,
        "Panel_Azimuth_deg": 180.0,
        "Plane_of_Array_Irradiance": 800.0,
        "Cell_Temperature_C": 35.0,
    }

    print("\nSample Input:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")

    # Create DataFrame
    input_df = pd.DataFrame([sample_data])[features]

    # Make prediction
    try:
        prediction = pipeline.predict(input_df)[0]
        print(f"\n✅ Prediction successful!")
        print(f"\n🌞 Predicted Power Output: {prediction:.2f} W")
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return

    # Load and display metrics
    print("\n📊 Model Performance Metrics:")
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        print(f"  Test MAE:  {metrics['test_metrics']['mae']} W")
        print(f"  Test RMSE: {metrics['test_metrics']['rmse']} W")
        print(f"  Test R²:   {metrics['test_metrics']['r2']}")

        print("\n🏆 Top 3 Important Features:")
        importance = metrics["feature_importance"]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_features[:3], 1):
            print(f"  {i}. {feat}: {imp:.4f}")
    else:
        print(f"  ⚠️ Metrics file not found: {metrics_path}")

    # Success message
    print("\n" + "=" * 70)
    print("🎉 All checks passed! Model is ready to use.")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start the API server: uvicorn src.server:app --reload")
    print("  2. Test the API: python src/test_api.py")
    print("  3. Launch dashboard: streamlit run src/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
