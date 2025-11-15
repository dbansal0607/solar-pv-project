"""
API Testing Script with Retry Logic
Tests the /predict endpoint with sample data

Usage:
    python src/test_api.py
"""

import requests
import json
import time

# API configuration
API_URL = "http://localhost:8000"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Sample test data
test_payload = {
    "Solar_Irradiance_kWh_m2": 0.85,
    "Temperature_C": 28.5,
    "Wind_Speed_mps": 4.2,
    "Relative_Humidity_%": 62.0,
    "Panel_Tilt_deg": 30.0,
    "Panel_Azimuth_deg": 180.0,
    "Plane_of_Array_Irradiance": 850.0,
    "Cell_Temperature_C": 38.0
}

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Health Check (GET /)")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print("✅ Health check successful!")
        print(f"\nAPI Status: {data['status']}")
        print(f"Model Version: {data['model_version']}")
        print(f"Features Required: {len(data['features_required'])}")
        print(f"\nResponse:")
        print(json.dumps(data, indent=2))
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused! Is the server running?")
        print("   Start it with: uvicorn src.server:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_prediction_with_retry():
    """Test the /predict endpoint with retry logic"""
    print("\n" + "="*70)
    print("TEST 2: Single Prediction (POST /predict)")
    print("="*70)
    
    print(f"\nTest Payload:")
    print(json.dumps(test_payload, indent=2))
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n🔄 Attempt {attempt}/{MAX_RETRIES}...")
            
            response = requests.post(
                f"{API_URL}/predict",
                json=test_payload,
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            print("✅ Prediction successful!")
            print(f"\n🌞 Predicted Power Output: {result['predicted_power']} W")
            print(f"Model Version: {result['model_version']}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"\nFull Response:")
            print(json.dumps(result, indent=2))
            return True
            
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection failed (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                print(f"   Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("   Max retries reached. Server may not be running.")
                return False
                
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error {e.response.status_code}")
            print(f"   Response: {e.response.text}")
            return False
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"   Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                return False
    
    return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 Solar PV API Testing Suite")
    print("="*70)
    print(f"API URL: {API_URL}")
    print(f"Max Retries: {MAX_RETRIES}")
    
    # Test 1: Health check
    health_ok = test_health_check()
    
    if not health_ok:
        print("\n" + "="*70)
        print("⚠️ TESTS ABORTED: Server not responding")
        print("="*70)
        print("\nTroubleshooting steps:")
        print("  1. Make sure you're in the project directory")
        print("  2. Activate the virtual environment:")
        print("     Windows: .\\venv\\Scripts\\Activate.ps1")
        print("     Linux/Mac: source venv/bin/activate")
        print("  3. Start the server:")
        print("     uvicorn src.server:app --reload --host 0.0.0.0 --port 8000")
        print("  4. Wait for 'Application startup complete' message")
        print("  5. Run this test again")
        return
    
    # Test 2: Prediction
    pred_ok = test_prediction_with_retry()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Prediction:   {'✅ PASS' if pred_ok else '❌ FAIL'}")
    print("="*70)
    
    if health_ok and pred_ok:
        print("\n🎉 All tests passed! API is working correctly.")
        print("\nNext steps:")
        print("  1. Run the Streamlit dashboard: streamlit run src/app.py")
        print("  2. Test batch predictions: python src/predict_unseen.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()