from firebase_functions import https_fn
from app.main import app

@https_fn.on_request()
def api(req: https_fn.Request) -> https_fn.Response:
    # This is a simplified wrapper. For full FastAPI support on Firebase, 
    # it's usually better to use the Firebase Gen 2 Python functions.
    # Note: Using Mangum or similar for full ASGI support is recommended.
    from mangum import Mangum
    handler = Mangum(app)
    return handler(req)
