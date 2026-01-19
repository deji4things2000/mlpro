"""
Quick API test script.
"""
import requests
import time

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing /health...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    print()

def test_upload():
    print("Testing file upload...")
    
    # Read medium.bin
    with open("../tests/test_data/medium.bin", "rb") as f:
        files = {"file": ("medium.bin", f, "application/octet-stream")}
        params = {"mode": "auto"}
        
        resp = requests.post(f"{BASE_URL}/api/analyze", files=files, params=params)
        
        if resp.status_code != 200:
            print(f"Upload failed: {resp.status_code}")
            print(resp.text)
            return None
            
        data = resp.json()
        print(f"Job started: {data['job_id']}")
        return data["job_id"]

def poll_results(job_id, max_attempts=30):
    print(f"Polling for results...")
    
    for i in range(max_attempts):
        time.sleep(1)
        
        resp = requests.get(f"{BASE_URL}/api/result/{job_id}")
        data = resp.json()
        
        print(f"Attempt {i+1}: {data['status']}")
        
        if data["status"] == "completed":
            print("\n=== ANALYSIS COMPLETE ===")
            print(f"Result: {data['result']}")
            
            if data["result"] and data["result"].get("solution"):
                sol = data["result"]["solution"]
                print(f"\n✅ SOLUTION FOUND!")
                print(f"   arg1: {sol['arg1']}")
                print(f"   arg2: {sol['arg2']}")
                print(f"\nRun: ./medium.bin '{sol['arg1']}' '{sol['arg2']}'")
            
            return data["result"]
        elif data["status"] == "error":
            print(f"Error: {data['error']}")
            return None
    
    print("Timeout waiting for results")
    return None

if __name__ == "__main__":
    print("="*60)
    print("RevCopilot Backend Test")
    print("="*60)
    
    test_health()
    
    job_id = test_upload()
    if job_id:
        poll_results(job_id)
