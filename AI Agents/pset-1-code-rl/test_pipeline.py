# test_pipeline.py
import os
import asyncio
import aiohttp

async def test_sandbox():
    """Test if sandbox is responding."""
    url = "http://localhost:8080/run_code"
    test_payload = {
        "code": "print('Hello, World!')",
        "language": "python"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=test_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ Sandbox is working: {result}")
                    return True
                else:
                    print(f"✗ Sandbox error: Status {response.status}")
                    return False
    except Exception as e:
        print(f"✗ Cannot connect to sandbox: {e}")
        return False

async def test_imports():
    """Test if all imports work."""
    try:
        import chz
        import tinker
        import torch
        import datasets
        import transformers
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

async def main():
    print("Testing pipeline setup...")
    
    print("\n1. Testing imports...")
    imports_ok = await test_imports()
    
    print("\n2. Testing sandbox connection...")
    sandbox_ok = await test_sandbox()
    
    if imports_ok and sandbox_ok:
        print("\n✅ All tests passed! You can run the training script.")
        print("\nRun: python train.py --max_steps 5 --batch_size 4 --group_size 2")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")

if __name__ == "__main__":
    asyncio.run(main())