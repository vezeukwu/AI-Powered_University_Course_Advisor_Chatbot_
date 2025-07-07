
import gradio as gr
import os
import zipfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline


# === 1. Extract FAISS Index if Needed ===
if not os.path.exists("faiss_index/index.faiss") and os.path.exists("faiss_index.zip"):
    with zipfile.ZipFile("faiss_index.zip", "r") as zip_ref:
        zip_ref.extractall()

assert os.path.exists("faiss_index/index.faiss"), "index.faiss not found after extraction."


# === 2. Load FAISS Vectorstore ===
print("🔧 Loading FAISS Vectorstore...")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore_faiss = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
print("✅ FAISS index loaded.")


# === 3. Load LLM (Flan-T5 Small) ===
print("🔧 Loading Language Model...")
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text_gen_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU-only for HF Spaces
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
print("✅ Language Model loaded.")


# === 4. Prompt and Response Generation ===
def format_prompt(context, question):
    return (
        "You are an assistant answering questions ONLY using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer_fn(question):
    docs = vectorstore_faiss.similarity_search(question, k=3)
    if not docs:
        return "I don't know based on the available information."
    
    context = "\n\n".join(d.page_content for d in docs)
    prompt = format_prompt(context, question)
    
    try:
        response = llm.invoke(prompt).strip()
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# === 5. Chatbot Function ===
def chat_response(message, chat_history):
    response = answer_fn(message)
    return response

# === 6. Gradio Chat Interface ===
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(
        fn=chat_response,
        title="🎓 Teesside Knowledge Base Chatbot",
        description="Ask about courses, tuition, departments, and more",
    )

# === 7. Launch ===
if __name__ == "__main__":
    demo.launch()
