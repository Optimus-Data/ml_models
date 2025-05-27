import os
from fastapi import HTTPException, status

def get_api_key() -> str:
    """
    Obtém a chave da API das variáveis de ambiente.
    Levanta um erro se a chave não estiver configurada.
    """
    api_key = os.getenv("ML_OPTIMUS_API_KEY")
    if not api_key:
        print("ERRO FATAL: ML_OPTIMUS_API_KEY não configurada no ambiente!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key do servidor não configurada."
        )
    return api_key
