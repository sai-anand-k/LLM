pip install streamlit transformers torch accelerate peft bitsandbytes trl huggingface_hub
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer

# Function to load the model
@st.cache_resource
def load_model():
    model_name = "aboonaji/llama2finetune-v2"
    llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                       quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                                             bnb_4bit_compute_dtype=getattr(torch, "float16"),
                                                                                             bnb_4bit_quant_type="nf4"))
    llama_model.config.use_cache = False
    llama_model.config.pretraining_tp = 1

    llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"
    
    text_generation_pipeline = pipeline(task="text-generation", model=llama_model, tokenizer=llama_tokenizer, max_length=300)
    return text_generation_pipeline, llama_tokenizer

# Load model and tokenizer
text_generation_pipeline, tokenizer = load_model()

# Streamlit app
st.title("Chatbot with Fine-Tuned LLaMA Model")

# Initialize chat history
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def generate_response(prompt):
    response = text_generation_pipeline(f"<s>[INST] {prompt} [/INST]")
    return response[0]['generated_text']

# Chatbot UI
user_input = st.text_input("You:", key="input")

if user_input:
    response = generate_response(user_input)

    # Store chat history
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.write(f"You: {st.session_state['past'][i]}")
        st.write(f"Bot: {st.session_state['generated'][i]}")

# To run the app, use the following command:
# streamlit run chatbot_app.py

