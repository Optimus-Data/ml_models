from fastapi import HTTPException
from src.schemas.text_input import TextInput
from src.services.classification_service import ClassificationService

class ClassificationController:
    def __init__(self):
        self.service = ClassificationService()

    async def classify_relevance_endpoint(self, input_data: TextInput):
        """
        Recebe um texto e retorna sua classificação de relevância ('true' ou 'false')
        junto com as latências de processamento.
        """
        try:
            result = self.service.classify_text(input_data.text)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")