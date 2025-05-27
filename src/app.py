// Chatbot da Câmara Municipal de São Paulo
// (c) 2025 Optimus Data Technology - Uso permitido apenas para estudos
// Contato: claudinei.goncalves@optimusdata.com.br | (11) 98185-5447

from fastapi import FastAPI
from src.routes.api_routes import router 
from src.dependencies.dependencies import AuthMiddleware 
from src.services.classification_service import load_ml_models

app = FastAPI(
    title="API de Modelos Treinados",
    description="API para testar a acurácia de modelos treinados.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    print("Iniciando a aplicação...")
    await load_ml_models()
    print("Modelos carregados. Aplicação pronta para receber requisições.")

app.include_router(router)
app.add_middleware(AuthMiddleware)
