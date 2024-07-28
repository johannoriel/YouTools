# YouTools
YouTube tools to automate video creation

## .llm-config.yml
```
model_list: 
  - model_name: groq1
    litellm_params:
      model: groq/llama3-70b-8192
      api_base: https://api.groq.com/openai/v1
      api_key: gsk_xxxxxxxxxxxx
  - model_name: ollama-phi3
    litellm_params:
      model: ollama/phi3:medium-128k
      api_base: http://127.0.0.1:11434

general_settings: 
  master_key: sk-xxxxx
```

## .env
GROQ optionnal
HF_TOKEN needed for SoniTranslate
```
YOUTUBE_API_KEY=xxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxx 
YOUR_HF_TOKEN=hf_xxxxxxxxxxx
```
## Install with pyenv
```
pyenv install 3.10
pyenv virtualenv 3.10 .venv
pyenv local .venv
pip install -r requirements.txt
./start.sh
```


