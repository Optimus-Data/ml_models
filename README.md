#  Logistic Relevance Classifier - Optimus Data Technology  

**Sistema de classificação de relevância baseado em Regressão Logística e embeddings multilíngues.**  

##  Visão Geral  
Este projeto implementa um classificador de relevância textual (`true`/`false`) usando:  
- **Sentence Transformers** (`distiluse-base-multilingual-cased-v2`) para geração de embeddings.  
- **Regressão Logística** (scikit-learn) para classificação.  
- **FastAPI** para expor um endpoint de inferência.

## 🛠 Stack Tecnológica
- **Linguagem**: Python 3.10+
- **Machine Learning**: Scikit-learn + Sentence-Transformers
- **Modelo de Embeddings**: distiluse-base-multilingual-cased-v2
- **Classificador**: Regressão Logística (LogisticRegression)
- **Serialização**: Joblib
- **API**: FastAPI + Uvicorn
- **Processamento NLP**: HuggingFace Transformers
- **Variáveis de Ambiente**: python-dotenv
- **Monitoramento**: Logs customizados + Colorama

##  Configuração  
1. **Variáveis de ambiente**: Crie um arquivo `.env` na raiz com:  
   ```plaintext
   LOGISTIC_DATA_PATH=caminho/para/seu/dataset.json
   Defina uma API KEY
   ```
   ## Instalação

```bash
pip install -r requirements.txt
```
## Treinamento

```bash
python logistic_regression.py
```
Métricas disponíveis no terminal após o treinamento


## 🔄 Requisição da API

### 📤 **Requisição (`POST /classify_relevance`)**
```json
{
  "text": "Insira seu texto aqui para classificação"
}

