import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to RevCopilot API"}

def test_some_endpoint():
    response = client.post("/some-endpoint", json={"key": "value"})
    assert response.status_code == 200
    assert "result" in response.json()