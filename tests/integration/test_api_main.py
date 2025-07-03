import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_predict_endpoint(client):
    # Example input matching your model's expected features
    payload = {
        "features": {
            "StdTxnValue": 0.0,
            "NumUniqueProducts": 1,
            "NumUniqueCategories": 1,
            "NumUniqueChannels": 1,
            "PreferredDayOfWeek": 2,
            "NetSpend": -10000.0,
            "GrossVolume": 10000,
            "TxnCount": 1,
            "AvgTxnValue": 10000.0,
            "Recency": 84,
            "PreferredProvider": "ProviderId_4",
            "MostCommonPricingStrategy": 4,
            "PreferredChannel": "ChannelId_2"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "probability" in response.json()
    assert isinstance(response.json()["probability"], float)