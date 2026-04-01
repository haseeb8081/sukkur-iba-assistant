import requests

def start_crawl():
    print("🚀 Triggering index for Sukkur IBA University (https://www.iba-suk.edu.pk/)")
    url = "http://localhost:8000/api/ingest/website"
    payload = {
        "url": "https://www.iba-suk.edu.pk/",
        "max_depth": 2
    }
    
    try:
        # Note: You must ensure uvicorn app.main:app is running first!
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Ingestion started successfully!")
            print(response.json())
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Connection error: Is your server running on port 8000?")

if __name__ == "__main__":
    start_crawl()
