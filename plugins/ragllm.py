from global_vars import translations, t
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

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "rag_plugin_loaded": "RAG LLM Plugin loaded",
    "rag_enter_text": "Enter RAG text:",
    "rag_enter_question": "Enter your question:",
    "rag_button_get_answer": "Get an answer",
    "rag_success_text_processed": "RAG text processed successfully!",
    "rag_warning_enter_text": "Please enter RAG text.",
    "rag_warning_process_text_first": "Please process the RAG text first.",
    "rag_warning_enter_question": "Please enter a question.",
    "rag_answer": "Answer:",
    "rag_citations": "Citations:",
    "rag_model_provider": "Model Provider",
    "rag_llm_model": "LLM Model",
    "rag_embedder_model": "Embedding Model",
    "rag_similarity_method": "Similarity Method",
    "rag_llm_sys_prompt": "System prompt for LLM",
    "rag_chunk_size": "Chunk size",
    "rag_top_k_chunks": "Number of chunks to use",
    "rag_default_sys_prompt": "You are an AI assistant. Your task is to analyze the provided context and answer questions based ONLY on this context. If the information is not in the context, clearly state that.",
    "rag_error_fetching_models_ollama": "Error fetching Ollama models: ",
    "rag_error_calling_llm": "Error calling LLM: "
})

translations["fr"].update({
    "rag_plugin_loaded": "Plugin RAG LLM chargé",
    "rag_enter_text": "Entrez le texte RAG :",
    "rag_enter_question": "Entrez votre question :",
    "rag_button_get_answer": "Obtenir une réponse",
    "rag_success_text_processed": "Texte RAG traité avec succès!",
    "rag_warning_enter_text": "Veuillez entrer du texte RAG.",
    "rag_warning_process_text_first": "Veuillez d'abord traiter le texte RAG.",
    "rag_warning_enter_question": "Veuillez entrer une question.",
    "rag_answer": "Réponse :",
    "rag_citations": "Citations :",
    "rag_model_provider": "Fournisseur de modèle",
    "rag_llm_model": "Modèle LLM",
    "rag_embedder_model": "Modèle d'embedding",
    "rag_similarity_method": "Méthode de similarité",
    "rag_llm_sys_prompt": "Prompt système pour le LLM",
    "rag_chunk_size": "Taille des chunks",
    "rag_top_k_chunks": "Nombre de chunks à utiliser",
    "rag_default_sys_prompt": "Tu es un assistant IA. Ta tâche est d'analyser le contexte fourni et de répondre aux questions en te basant UNIQUEMENT sur ce contexte. Si l'information n'est pas dans le contexte, dis-le clairement.",
    "rag_error_fetching_models_ollama": "Erreur lors de la récupération des modèles Ollama : ",
    "rag_error_calling_llm": "Erreur lors de l'appel au LLM : "
})

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
                "label": t("rag_model_provider"),
                "options": [("ollama", "Ollama"), ("groq", "Groq")],
                "default": "ollama"
            },
            "llm_model": {
                "type": "select",
                "label": t("rag_llm_model"),
                "options": [("none", "À charger...")],
                "default": "ollama/qwen2"
            },
            "embedder": {
                "type": "select",
                "label": t("rag_embedder_model"),
                "options": [
                    ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
                    ("nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text-v1.5")
                ],
                "default": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "similarity_method": {
                "type": "select",
                "label": t("rag_similarity_method"),
                "options": [
                    ("cosine", "Cosinus"),
                    ("euclidean", "Distance euclidienne"),
                    ("manhattan", "Distance de Manhattan")
                ],
                "default": "cosine"
            },
            "llm_sys_prompt": {
                "type": "textarea",
                "label": t("rag_llm_sys_prompt"),
                "default": t("rag_default_sys_prompt")
            },
            "chunk_size": {
                "type": "number",
                "label": t("rag_chunk_size"),
                "default": 200
            },
            "top_k": {
                "type": "number",
                "label": t("rag_top_k_chunks"),
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
                st.error(f"{t('rag_error_fetching_models_ollama')}{str(e)}")
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
            return f"{t('rag_error_calling_llm')}{str(e)}"

    def run(self, config):
        st.write(t("rag_plugin_loaded"))
       
        rag_text = st.text_area(t("rag_enter_text"), height=200)
        user_prompt = st.text_area(t("rag_enter_question"), "résume")

        if st.button(t("rag_button_get_answer")):
            if rag_text:
                self.process_rag_text(rag_text, config['ragllm']['chunk_size'], config['ragllm']['embedder'])
                st.success(t("rag_success_text_processed"))
            else:
                st.warning(t("rag_warning_enter_text"))        
            if user_prompt and self.embeddings is not None:
                context, citations = self.obtenir_contexte(user_prompt, config)
                response = self.process_with_llm(user_prompt, config['ragllm']['llm_sys_prompt'], context, config['ragllm']['llm_model'])
                
                st.write(t("rag_answer"))
                st.write(response)
                
                st.write(t("rag_citations"))
                for i, citation in enumerate(citations, 1):
                    st.write(f"{i}. {citation[:100]}...")
            elif self.embeddings is None:
                st.warning(t("rag_warning_process_text_first"))
            else:
                st.warning(t("rag_warning_enter_question"))

