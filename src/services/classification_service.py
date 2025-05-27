import time
import numpy as np
from sentence_transformers import SentenceTransformer
from joblib import load
import torch 
import os 

classifier_model_global = None
embeddings_model_global = None
inference_device_global = "cpu"

class ClassificationService:
    def __init__(self):
        if classifier_model_global is None or embeddings_model_global is None:
            raise RuntimeError("Modelos de ML não carregados. A API não está pronta.")
        self.classifier_model = classifier_model_global
        self.embeddings_model = embeddings_model_global
        self.device = inference_device_global

    def classify_text(self, text: str):
        overall_start_time = time.time()

        embedding_start_time = time.time()
        embedding_texto = self.embeddings_model.encode(
            [text], show_progress_bar=False, convert_to_tensor=True
        )
        embedding_end_time = time.time()
        embedding_latency = embedding_end_time - embedding_start_time

        prediction_start_time = time.time()
        previsao_numerica = self.classifier_model.predict(
            embedding_texto.cpu().numpy()
        )[0]
        prediction_end_time = time.time()
        prediction_latency = prediction_end_time - prediction_start_time

        label = "true" if previsao_numerica == 1 else "false"
        overall_end_time = time.time()
        overall_latency = overall_end_time - overall_start_time

        return {
            "query": text,
            "relevance": label,
            "latencies": {
                "embedding_generation_seconds": embedding_latency,
                "classification_seconds": prediction_latency,
                "total_inference_seconds": overall_latency
            }
        }

async def load_ml_models():
    global classifier_model_global, embeddings_model_global, inference_device_global
    load_start_time = time.time()

    if torch.cuda.is_available():
        print(f"GPU (CUDA/ROCm) detectada! Usando: {torch.cuda.get_device_name(0)}")
        inference_device_global = "cuda"
    else:
        print("GPU não detectada ou PyTorch não configurado para GPU. Usando CPU.")
        inference_device_global = "cpu"

    try:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 
            '..', 
            'trained_models',
            'classifier_model.joblib'
        )
        
        classifier_model_global = load(model_path) 
        print(f"Modelo de classificação carregado com sucesso de: {model_path}")
    except FileNotFoundError:
        print(Fore.RED + f"ERRO FATAL: '{model_path}' não encontrado. Certifique-se de que o modelo foi treinado e salvo.")
        raise RuntimeError(f"Modelo de classificação não encontrado. A API não pode iniciar.")
    except Exception as e:
        print(f"ERRO FATAL ao carregar o classificador: {e}")
        raise RuntimeError(f"Falha ao carregar o classificador: {e}")

    try:
        embeddings_model_global = SentenceTransformer('distiluse-base-multilingual-cased-v2', device=inference_device_global)
        print("Modelo de embeddings carregado com sucesso!")
    except Exception as e:
        print(f"ERRO FATAL ao carregar o modelo de embeddings: {e}")
        raise RuntimeError(f"Falha ao carregar o modelo de embeddings: {e}")

    load_end_time = time.time()
    print(f"Tempo total para carregar os modelos: {load_end_time - load_start_time:.4f} segundos\n")