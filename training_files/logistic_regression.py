from colorama import init, Fore, Style
init(autoreset=True)  

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import json
import os
from joblib import dump, load
import time
from dotenv import load_dotenv 

load_dotenv()

# ╔══════════════════════════════════════════════════════════════════════╗
# ║        [SYSTEM BOOT] >> INITIATING MACHINE LEARNING SEQUENCE        ║
# ╚══════════════════════════════════════════════════════════════════════╝

print(Fore.CYAN + Style.BRIGHT + "\n>>> [BOOT] Sistema de Machine Learning Iniciado...\n")

# ──[ CORE MODULE: DATA ACQUISITION ]─────────────────────────────────────
start_time = time.time()
data_file_path = os.getenv("LOGISTIC_DATA_PATH") 

if not data_file_path:
    print(Fore.RED + "[✖ ERROR::CODE-ENV] Variável de ambiente 'TRAINING_DATA_PATH' não definida no .env")
    exit()

try:
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(Fore.GREEN + f"[✓] Data uplink established → {data_file_path}")
except FileNotFoundError:
    print(Fore.RED + f"[✖ ERROR::CODE-404] Data file not found → {data_file_path}")
    exit()
except json.JSONDecodeError:
    print(Fore.RED + f"[✖ ERROR::CODE-500] Invalid JSON syntax in → {data_file_path}")
    exit()

df = pd.DataFrame(data)
df['label_encoded'] = df['label'].apply(lambda x: 1 if x == 'true' else 0)

X = df['text']
y = df['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(Fore.YELLOW + f"[DATA STATUS] Train={len(X_train)} | Test={len(X_test)}")
print(Fore.MAGENTA + f"[SYSTEM TIME] Data split complete in {time.time() - start_time:.2f}s")

# ──[ MODULE: EMBEDDING GENERATOR ]──────────────────────────────────────
start_time_embeddings = time.time()
print(Fore.CYAN + "\n[LOADING] Carregando motor de embeddings multilíngue...")

model_embeddings = SentenceTransformer('distiluse-base-multilingual-cased-v2')

print(Fore.CYAN + "[ENCODE] Vetorizando os dados de treino...")
X_train_embeddings = model_embeddings.encode(X_train.tolist(), show_progress_bar=True)

print(Fore.CYAN + "[ENCODE] Vetorizando os dados de teste...")
X_test_embeddings = model_embeddings.encode(X_test.tolist(), show_progress_bar=True)

print(Fore.MAGENTA + f"[COMPLETE] Embeddings gerados em {time.time() - start_time_embeddings:.2f}s")

# ──[ MODULE: MODEL TRAINING PROTOCOL ]──────────────────────────────────
start_time_train = time.time()
print(Fore.CYAN + "\n[TRAINING] Treinando modelo de Regressão Logística...")

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_embeddings, y_train)

print(Fore.GREEN + f"[✓ COMPLETE] Treinamento finalizado em {time.time() - start_time_train:.2f}s")

# ──[ MODULE: EVALUATION SEQUENCE ]──────────────────────────────────────
start_time_eval = time.time()
print(Fore.CYAN + "\n[EVALUATING] Executando diagnóstico do modelo...")

y_pred = classifier.predict(X_test_embeddings)

print(Fore.BLUE + "\n=== ✦ RELATÓRIO DE CLASSIFICAÇÃO ✦ ===")
print(Fore.WHITE + classification_report(y_test, y_pred, target_names=['false', 'true']))
print(Fore.YELLOW + f"[METRIC] Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(Fore.MAGENTA + f"[SYSTEM TIME] Avaliação concluída em {time.time() - start_time_eval:.2f}s")

# ──[ MODULE: SAVE STATE ]───────────────────────────────────────────────
start_time_save = time.time()
model_dir = os.path.join(os.path.dirname(__file__), '..', 'trained_models')
os.makedirs(model_dir, exist_ok=True) 

model_path = os.path.join(model_dir, 'classifier_model.joblib')
dump(classifier, model_path)
print(Fore.GREEN + f"\n[✓ SAVE] Modelo salvo como → '{model_path}'")
print(Fore.MAGENTA + f"[SYSTEM TIME] Duração da gravação: {time.time() - start_time_save:.2f}s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                SYSTEM STANDBY FOR NEXT EXECUTION                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
print(Fore.CYAN + Style.BRIGHT + "\n>>> [STANDBY] Missão concluída. Sistema aguardando nova tarefa...\n")