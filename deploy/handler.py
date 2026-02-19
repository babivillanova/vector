"""AWS Lambda entry point — adapts FastAPI ASGI app to Lambda events."""

from mangum import Mangum
from api import app

handler = Mangum(app, lifespan="off")
