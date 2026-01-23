import requests
import time

print("=== RevCopilot Backend Test ===")

# 1. Test health
print("\n1. Testing /health...")
resp = requests.get("http://localhost:8000/health")
print(f"   Status: {resp.status_code}")
print(f"   Response: {resp.json()}")

# 2. Test solution endpoint
print("\n2. Testing /test/solution...")
resp = requests.get("http://localhost:8000/test/solution")
print(f"   Status: {resp.status_code}")
print(f"   Solution: {resp.json()['solution']}")

# 3. Upload medium.bin
print("\n3. Uploading medium.bin...")
with open("../tests/test_data/medium.bin", "rb") as f:
    files = {"file": ("medium.bin", f, "application/octet-stream")}
    resp = requests.post("http://localhost:8000/api/analyze?mode=auto", files=files)

if resp.status_code == 200:
    data = resp.json()
    job_id = data["job_id"]
    print(f"   Job created: {job_id}")
    
    # 4. Wait and get results
    print("\n4. Waiting for results...")
    for i in range(5):
        time.sleep(1)
        result_resp = requests.get(f"http://localhost:8000/api/result/{job_id}")
        
        if result_resp.status_code == 200:
            result_data = result_resp.json()
            status = result_data.get("status", "unknown")
            print(f"   Attempt {i+1}: Status = {status}")
            
            if status == "completed":
                print("\n   === ANALYSIS COMPLETE ===")
                result = result_data.get("result", {})
                
                if result.get("solution"):
                    sol = result["solution"]
                    print(f"   Solution found!")
                    print(f"   arg1: {sol.get('arg1')}")
                    print(f"   arg2: {sol.get('arg2')}")
                else:
                    print(f"   Result: {result}")
                break
            elif status == "error":
                print(f"   Error: {result_data.get('error')}")
                break
    else:
        print("   Timeout after 5 seconds")
else:
    print(f"   Upload failed: {resp.status_code}")
    print(f"   Response: {resp.text}")

print("\n=== Test Complete ===")
