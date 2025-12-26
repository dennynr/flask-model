"""
Flask API Service untuk Klasifikasi Kualitas Air
Persistent service yang load model sekali saja, jauh lebih cepat dari shell_exec

Install requirements:
pip install flask flask-cors pandas joblib scikit-learn numpy

Jalankan:
python ClassificationAPI.py

API akan berjalan di http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model sekali saja saat startup (bukan setiap request!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model_decision_tree_lobster.pkl')
MODEL_PATH = os.path.normpath(MODEL_PATH)

print(f"Loading model from: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def simple_classification(ph, tds, suhu, do):
    """Fallback classification jika model tidak tersedia"""
    classification = 1  # Default: Layak
    reasons = []
    not_suitable_count = 0
    less_suitable_count = 0

    # Cek pH
    if ph < 6.0 or ph > 8.5:
        not_suitable_count += 1
        reasons.append(f'pH tidak layak ({ph}) - Range layak: 6.5-7.8')
    elif (6.0 <= ph < 6.5) or (7.8 < ph <= 8.5):
        less_suitable_count += 1
        reasons.append(f'pH kurang layak ({ph}) - Range layak: 6.5-7.8')

    # Cek TDS
    if tds > 600:
        not_suitable_count += 1
        reasons.append(f'TDS tidak layak ({tds} mg/L) - Range layak: 50-400 mg/L')
    elif (tds < 50) or (400 < tds <= 600):
        less_suitable_count += 1
        reasons.append(f'TDS kurang layak ({tds} mg/L) - Range layak: 50-400 mg/L')

    # Cek Suhu
    if suhu < 21 or suhu > 27:
        not_suitable_count += 1
        reasons.append(f'Suhu tidak layak ({suhu}°C) - Range layak: 23-25°C')
    elif (21 <= suhu < 23) or (25 < suhu <= 27):
        less_suitable_count += 1
        reasons.append(f'Suhu kurang layak ({suhu}°C) - Range layak: 23-25°C')

    # Cek DO
    if do < 2.5 or do > 7:
        not_suitable_count += 1
        reasons.append(f'DO tidak layak ({do} mg/L) - Range layak: 4-6 mg/L')
    elif (2.5 <= do < 4) or (6 < do <= 7):
        less_suitable_count += 1
        reasons.append(f'DO kurang layak ({do} mg/L) - Range layak: 4-6 mg/L')
    
    # Tentukan klasifikasi
    if not_suitable_count > 0:
        classification = 2  # Tidak Layak
    elif less_suitable_count > 0:
        classification = 0  # Kurang Layak
    else:
        classification = 1  # Layak
    
    label_names = {0: 'Kurang Layak', 1: 'Layak', 2: 'Tidak Layak'}
    
    return {
        'classification': classification,
        'classification_label': label_names.get(classification, 'Unknown'),
        'confidence': None,
        'method': 'simple_threshold',
        'reasons': reasons,
        'not_suitable_count': not_suitable_count,
        'less_suitable_count': less_suitable_count
    }

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Classification API is running'
    })

@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify water quality
    
    POST /classify
    {
        "ph": 7.0,
        "tds": 200,
        "suhu": 24,
        "do": 5.0
    }
    """
    try:
        data = request.get_json()
        
        # Validasi input
        if not all(k in data for k in ['ph', 'tds', 'suhu', 'do']):
            return jsonify({
                'error': 'Missing required parameters. Need: ph, tds, suhu, do'
            }), 400
        
        ph = float(data['ph'])
        tds = float(data['tds'])
        suhu = float(data['suhu'])
        do_value = float(data['do'])
        
        # Jika model tidak tersedia, gunakan simple classification
        if model is None:
            result = simple_classification(ph, tds, suhu, do_value)
            result['note'] = 'Using fallback classification (model not loaded)'
            return jsonify(result)
        
        # Prediksi menggunakan model
        df = pd.DataFrame([{
            "suhu": suhu,
            "ph": ph,
            "do": do_value,
            "tds": tds
        }])
        
        prediction = model.predict(df)[0]
        
        # Convert ke integer jika perlu
        if isinstance(prediction, str):
            label_map = {
                'Layak': 1,
                'Kurang Layak': 0,
                'Kurang layak': 0,
                'Tidak Layak': 2,
                'Tidak layak': 2
            }
            prediction_int = label_map.get(prediction.strip(), 1)
        else:
            prediction_int = int(prediction)
        
        # Get probability jika tersedia
        try:
            probability = model.predict_proba(df)[0]
            confidence = float(max(probability) * 100)
        except:
            confidence = None
        
        label_names = {0: 'Kurang Layak', 1: 'Layak', 2: 'Tidak Layak'}
        
        result = {
            'classification': prediction_int,
            'classification_label': label_names.get(prediction_int, 'Unknown'),
            'confidence': confidence,
            'method': 'decision_tree_model',
            'original_prediction': str(prediction),
            'input': {
                'ph': ph,
                'tds': tds,
                'suhu': suhu,
                'do': do_value
            }
        }
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Water Quality Classification API")
    print("="*60)
    print("API running on: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health    - Health check")
    print("  POST /classify  - Classify water quality")
    print("\nExample request:")
    print('  POST http://localhost:5000/classify')
    print('  {"ph": 7.0, "tds": 200, "suhu": 24, "do": 5.0}')
    print("="*60 + "\n")
    
    app.run()

