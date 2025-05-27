// Chatbot da Câmara Municipal de São Paulo
// (c) 2025 Optimus Data Technology - Uso permitido apenas para estudos
// Contato: claudinei.goncalves@optimusdata.com.br | (11) 98185-5447

import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

from src.app import app

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
