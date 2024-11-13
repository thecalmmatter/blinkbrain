import streamlit as st
import vid2cleantxt
import ollama
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import uuid

# Initialize embedding model
@st.cache_resource
def initialize_embedder():
    #return SentenceTransformer('all-MiniLM-L6-v2')
    return SentenceTransformer('all-distilroberta-v1') 

embedder = initialize_embedder()

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Get or create the transcription_embeddings collection
def get_or_create_collection(client, collection_name):
    try:
        return client.create_collection(collection_name)
    except chromadb.errors.UniqueConstraintError:
        return client.get_collection(collection_name)

collection_name = "transcription_embeddings"
collection = get_or_create_collection(chroma_client, collection_name)

# Title and description
st.title("🎬 Enhanced Video Transcription & Interactive Q&A App with Llama")
st.write("Upload a video, transcribe it, fine-tune the text using Llama, and interact with the transcription through Q&A.")


# Function to fine-tune transcription text with Llama API via Ollama
def call_llama_finetune(text: str) -> str:
    prompt = (
        "Pretend to be a professional writer and finetune the text, making appropriate punctuation adjustments. "
        "Don't change the dialogue or any words as the text is directly generated from video and is translated; "
        "however, check for any grammatical mistakes and basic grammar.\n\n" + text
    )
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
    refined_text = response['message']['content']
    return refined_text


# Function to add embeddings to ChromaDB
def add_embeddings_to_chromadb(sentences):
    embeddings = embedder.encode(sentences)
    ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
    for i, embedding in enumerate(embeddings):
        collection.add(
            documents=[sentences[i]],
            embeddings=[embedding.tolist()],
            metadatas=[{"sentence_index": i}],
            ids=[ids[i]]
        )

# Function to retrieve relevant context from ChromaDB
def get_relevant_context_from_chromadb(query, top_k=3):
    query_embedding = embedder.encode(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    relevant_sentences = [result for doc in results['documents'] for result in doc]
    return "\n".join(relevant_sentences)

# Video file uploader
st.subheader("📁 Upload Your Video File")
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# Only process if the video file is uploaded
if video_file is not None:
    video_path = Path(video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    st.success(f"✅ Video file '{video_file.name}' uploaded successfully!")
    st.video(video_file)  # Show video in Streamlit

    # Transcribe video
    with st.spinner("Transcribing video... ⏳"):
        text_output, metadata_output = vid2cleantxt.transcribe.transcribe_dir(
            input_dir=".",
            model_id='openai/whisper-small.en',
            chunk_length=30,
            join_text=True
        )

    st.success("🎉 Transcription completed successfully!")

    # Display the transcription text
    st.subheader("📜 Original Transcribed Text")
    output_dir = Path(text_output)
    files = [f for f in output_dir.iterdir() if f.suffix == '.txt']
    with open(files[0], 'r') as f:
        transcribed_text = f.read()
    st.text_area("Transcribed Text", transcribed_text, height=300)

    # Fine-tune transcription text with Llama
    st.subheader("✍️ Fine-tuning Transcription Text")
    with st.spinner("Fine-tuning the transcribed text with Llama... ⏳"):
        fine_tuned_text = call_llama_finetune(transcribed_text)

    st.success("✨ Fine-tuning completed!")
    st.subheader("📃 Fine-tuned Text")
    st.text_area("Fine-tuned Transcription", fine_tuned_text, height=300)

    # Generate and store embeddings in ChromaDB
    with st.spinner("Storing embeddings for Q&A in ChromaDB..."):
        sentences = [sentence.strip() for sentence in fine_tuned_text.split(". ") if sentence]
        add_embeddings_to_chromadb(sentences)
    st.success("Embeddings stored in ChromaDB!")

    # Q&A Section
    st.subheader("💬 Ask Questions Based on the Transcription")
    user_question = st.text_input("Enter your question:")

    # Process query only when the user clicks "Submit" button
    if st.button("Submit Question") and user_question:
        with st.spinner("Retrieving relevant context and generating answer..."):
            # Retrieve relevant context from ChromaDB
            relevant_context = get_relevant_context_from_chromadb(user_question)
            
            # Generate response with context-based prompt to Llama
            query = f"""Use the following information to answer the question. If the answer cannot be found, write "I don't know."
            Context:
            \"\"\"
            {relevant_context}
            \"\"\"
            Question: {user_question}"""

            response = ollama.chat(model='llama3.1', messages=[
                {"role": "system", "content": "You are an educational assistant."},
                {"role": "user", "content": query},
            ])
            answer = response['message']['content']

        # Display the answer and context used
        st.write("**Assistant's Response:**")
        st.write(answer)
        st.write("**Relevant Context Pulled from Transcription:**")
        st.write(relevant_context)

    # Option to download the fine-tuned transcription
    st.download_button(
        label="💾 Download Fine-tuned Transcription",
        data=fine_tuned_text,
        file_name="fine_tuned_transcription.txt",
        mime="text/plain"
    )

    # Remove uploaded video file
    if video_path.exists():
        video_path.unlink()
        st.success("🗑️ Uploaded video file has been deleted.")
else:
    st.info("Please upload a video file to start the transcription process.")

# Footer
st.markdown(
    """
    ---
    ### ℹ️ About This App
    This app uses the [vid2cleantxt](https://github.com/pszemraj/vid2cleantxt) package for transcription and Ollama's Llama model for fine-tuning and interactive Q&A with ChromaDB.
    Developed by [Your Name](https://github.com/yourprofile).
    """
)