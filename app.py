print("App is starting...")
import gradio as gr
from pdf_utils import extract_text_from_pdf
from qa_engine import load_model, create_vector_store, get_answer
from summarizer import summarize_text

# Load models and tokenizer with the updated load_model function
model, tokenizer, embedding_model = load_model()

# Handle PDF upload, extract text, and create a vector store
def handle_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    vector_store = create_vector_store(text, embedding_model)
    return "PDF loaded successfully!", text, vector_store

# Handle question answering
def ask_question(vector_store, question):
    if not question.strip():
        return "Please enter a valid question."
    return get_answer(model, tokenizer, vector_store, question)

# Handle PDF summarization
def summarize(vector_store):
    return summarize_text(vector_store)

# Gradio UI setup
with gr.Blocks() as demo:
    gr.Markdown("# 🐼 PDF Whisperer – Ask and Summarize PDFs")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
        load_btn = gr.Button("Load PDF")
    
    output_status = gr.Textbox(label="Status")
    pdf_text = gr.Textbox(label="Extracted Text", lines=10)
    vector_store_state = gr.State()

    with gr.Row():
        question = gr.Textbox(label="Ask a question")
        ask_btn = gr.Button("Get Answer")

    answer_output = gr.Textbox(label="Answer", lines=5)
    
    with gr.Row():
        summary_btn = gr.Button("Summarize PDF")
        summary_output = gr.Textbox(label="Summary", lines=6)

    # Event handling for the Gradio interface
    load_btn.click(fn=handle_pdf, inputs=pdf_input, outputs=[output_status, pdf_text, vector_store_state])
    ask_btn.click(fn=ask_question, inputs=[vector_store_state, question], outputs=answer_output)
    summary_btn.click(fn=summarize, inputs=vector_store_state, outputs=summary_output)

# Launch Gradio interface
demo.launch()