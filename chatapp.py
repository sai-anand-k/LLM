pip install streamlit transformers torch accelerate peft bitsandbytes trl huggingface_hub
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the fine-tuned model and tokenizer
model_name = "aboonaji/llama2finetune-v2"  # Adjust according to your notebook
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True, 
                                                                                   bnb_4bit_compute_dtype=getattr(torch, "float16"), 
                                                                                   bnb_4bit_quant_type="nf4"))
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Initialize text generation pipeline
text_generation_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=300)

# Streamlit app
st.title("ChatGPT with Fine-Tuned LLaMA Model")

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
# streamlit run your_script.py
