from fastapi import APIRouter, Depends
from src.schemas.text_input import TextInput
from src.controllers.classification_controller import ClassificationController
from src.dependencies.dependencies import verify_api_key

router = APIRouter()

def get_classification_controller():
    return ClassificationController()

@router.post("/classify_relevance", dependencies=[Depends(verify_api_key)])
async def classify_relevance(
    input_data: TextInput,
    controller: ClassificationController = Depends(get_classification_controller) 
):
    return await controller.classify_relevance_endpoint(input_data)


    