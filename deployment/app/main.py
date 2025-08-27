from fastapi import FastAPI
from database import Base, engine
from routers import (
                     auth,bank_data, 
                     general_knowledge_extraction,
                     model, prediction)

from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="My API",
    version="1.0.0",
    description="API with JWT Authentication and Swagger"
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(auth.router)
app.include_router(bank_data.router)
app.include_router(general_knowledge_extraction.router)
app.include_router(model.router)
app.include_router(prediction.router)

# Custom OpenAPI schema to support Bearer token in Swagger
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="My API",
        version="1.0.0",
        description="API with JWT Authentication and Swagger",
        routes=app.routes,
    )
    # Enhance the existing security scheme (no need to add a new one)
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}
    openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi