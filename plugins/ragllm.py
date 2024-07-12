from app import Plugin
import streamlit as st
import yaml
from litellm import completion, embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import os
from typing import List, Dict, Any
import requests
import torch
from transformers import AutoTokenizer, AutoModel
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 512
CHUNK_SIZE = 200  # Nombre de mots par chunk

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class RagllmPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.config = self.load_llm_config()
        self.embeddings = None
        self.chunks = None

    def load_llm_config(self) -> Dict:
        with open('.llm-config.yml', 'r') as file:
            return yaml.safe_load(file)

    def get_tabs(self):
        return [{"name": "RAG", "plugin": "ragllm"}]
        
    def get_config_fields(self):
        return {
            "provider": {
                "type": "select",
                "label": "Fournisseur de modèle",
                "options": [("ollama", "Ollama"), ("groq", "Groq")],
                "default": "ollama"
            },
            "llm_model": {
                "type": "select",
                "label": "Modèle LLM",
                "options": [("none", "À charger...")],
                "default": "ollama/qwen2"
            },
            "embedder": {
                "type": "select",
                "label": "Modèle d'embedding",
                "options": [
                    ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
                    ("nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text-v1.5")
                ],
                "default": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "similarity_method": {
                "type": "select",
                "label": "Méthode de similarité",
                "options": [
                    ("cosine", "Cosinus"),
                    ("euclidean", "Distance euclidienne"),
                    ("manhattan", "Distance de Manhattan")
                ],
                "default": "cosine"
            },
            "llm_sys_prompt": {
                "type": "textarea",
                "label": "Prompt système pour le LLM",
                "default": "Tu es un assistant IA. Ta tâche est d'analyser le contexte fourni et de répondre aux questions en te basant UNIQUEMENT sur ce contexte. Si l'information n'est pas dans le contexte, dis-le clairement."
            },
            "chunk_size": {
                "type": "number",
                "label": "Taille des chunks",
                "default": 200
            },
            "top_k": {
                "type": "number",
                "label": "Nombre de chunks à utiliser",
                "default": 3
            }
        }

    def get_config_ui(self, config):
        updated_config = {}
        for field, params in self.get_config_fields().items():
            if params['type'] == 'select':
                if field == 'llm_model':
                    provider = config.get('provider', 'ollama')
                    models = self.get_available_models(provider)
                    updated_config[field] = st.selectbox(
                        params['label'],
                        options=models,
                        index=models.index(config.get(field, params['default']))
                    )
                else:
                    updated_config[field] = st.selectbox(
                        params['label'],
                        options=[option[0] for option in params['options']],
                        format_func=lambda x: dict(params['options'])[x],
                        index=[option[0] for option in params['options']].index(config.get(field, params['default']))
                    )
            elif params['type'] == 'textarea':
                updated_config[field] = st.text_area(
                    params['label'],
                    value=config.get(field, params['default'])
                )
            elif params['type'] == 'number':
                updated_config[field] = st.number_input(
                    params['label'],
                    value=int(config.get(field, params['default'])),
                    step=1
                )
            else:
                updated_config[field] = st.text_input(
                    params['label'],
                    value=config.get(field, params['default'])
                )
        return updated_config

    def get_available_models(self, provider: str) -> List[str]:
        if provider == 'ollama':
            try:
                response = requests.get("http://localhost:11434/api/tags")
                models = response.json()["models"]
                return [f"ollama/{model['name']}" for model in models] + ["ollama/qwen2"]
            except Exception as e:
                st.error(f"Erreur lors de la récupération des modèles Ollama : {str(e)}")
                return ["ollama/qwen2"]
        elif provider == 'groq':
            return ["groq/llama3-70b-8192", "groq/mixtral-8x7b-32768"]
        else:
            return ["none"]

    def process_rag_text(self, rag_text: str, chunk_size: int, embedder):
        rag_text = rag_text.replace('\\n', ' ').replace('\\\'', "'")
        mots = rag_text.split()
        self.chunks = [' '.join(mots[i:i+chunk_size]) for i in range(0, len(mots), chunk_size)]
        self.embeddings = np.vstack([self.get_embedding(c, embedder) for c in self.chunks])

    def get_embedding(self, text: str, model: str) -> np.ndarray:
        #response = embedding(model=model, input=[text])
        #return np.array(response['data'][0]['embedding'])
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model, trust_remote_code=True).to(DEVICE)
        inputs = tokenizer(text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model_output = model(**inputs)
        return mean_pooling(model_output, inputs['attention_mask']).cpu().numpy()

    def calculate_similarity(self, query_embedding: np.ndarray, method: str) -> np.ndarray:
        if method == 'cosine':
            return cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        elif method == 'euclidean':
            return -euclidean_distances(query_embedding.reshape(1, -1), self.embeddings)[0]
        elif method == 'manhattan':
            return -manhattan_distances(query_embedding.reshape(1, -1), self.embeddings)[0]
        else:
            raise ValueError("Méthode de similarité non reconnue")

    def obtenir_contexte(self, query: str, config: Dict[str, Any]) -> tuple:
        query_embedding = self.get_embedding(query, config['ragllm']['embedder'])
        similarities = self.calculate_similarity(query_embedding, config['ragllm']['similarity_method'])
        top_indices = np.argsort(similarities)[-config['ragllm']['top_k']:][::-1]
        context = "\n\n".join([self.chunks[i] for i in top_indices])
        return context, [self.chunks[i] for i in top_indices]

    def process_with_llm(self, prompt: str, sysprompt: str, context: str, llm_model: str) -> str:
        try:
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": f"Contexte : {context}\n\nQuestion : {prompt}"}
            ]
            response = completion(model=llm_model, messages=messages)
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Erreur lors de l'appel au LLM : {str(e)}"

    def run(self, config):
        st.write("Plugin RAG LLM chargé")
       
        rag_text = st.text_area("Entrez le texte RAG :", height=200)
        user_prompt = st.text_area("Entrez votre question :", "résume")

        if st.button("Obtenir une réponse"):
            if rag_text:
                self.process_rag_text(rag_text, config['ragllm']['chunk_size'], config['ragllm']['embedder'])
                st.success("Texte RAG traité avec succès!")
            else:
                st.warning("Veuillez entrer du texte RAG.")        
            if user_prompt and self.embeddings is not None:
                context, citations = self.obtenir_contexte(user_prompt, config)
                response = self.process_with_llm(user_prompt, config['ragllm']['llm_sys_prompt'], context, config['ragllm']['llm_model'])
                
                st.write("Réponse :")
                st.write(response)
                
                st.write("Citations :")
                for i, citation in enumerate(citations, 1):
                    st.write(f"{i}. {citation[:100]}...")
            elif self.embeddings is None:
                st.warning("Veuillez d'abord traiter le texte RAG.")
            else:
                st.warning("Veuillez entrer une question.")
