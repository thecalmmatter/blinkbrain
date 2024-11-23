# **Blink Brain - Enhanced Video Transcription & Interactive Q&A App with Llama** 🎬


![blinkbrain](https://github.com/user-attachments/assets/e1d05a2d-cf67-4216-9c1f-fe6ea3d67435)


This project is an application that allows users to upload a video, transcribe it, fine-tune the transcription using Llama, and interact with the transcription via a Q&A feature.

---

## **1. Project Overview**

This app is designed to assist users in extracting and improving video transcriptions with the power of **Llama**, a language model, and **ChromaDB** for efficient document management. The application uses **vid2cleantxt** for transcription and **SentenceTransformers** to create embeddings, which can be queried for context-based responses.

---

## **2. Features**

- **Video Transcription**: Upload a video and transcribe its audio to text.
- **Fine-Tuning**: The transcribed text is fine-tuned using the Llama model via Ollama for grammar and readability.
- **Q&A**: Users can ask questions based on the fine-tuned transcription, with context being retrieved from the embedded sentences.
- **Embedding Storage**: The transcriptions are stored in **ChromaDB** as embeddings, enabling efficient semantic search.

---

## **3. Functions**

### **`initialize_embedder()` Function**
   **Purpose**: Initializes the SentenceTransformer model for embedding text.  
   **Details**: Uses the `'all-distilroberta-v1'` model for generating sentence embeddings, cached for efficient re-use.  
   **Application**: Text embedding for downstream tasks like semantic search and clustering.

### **`get_or_create_collection()` Function**
   **Purpose**: Handles collection creation or retrieval from ChromaDB.  
   **Details**: Ensures that the ChromaDB collection for storing text embeddings exists.  
   **Application**: Used to manage embeddings in ChromaDB for later retrieval.

### **`call_llama_finetune()` Function**
   **Purpose**: Fine-tunes the transcribed text using the Llama model via the Ollama API.  
   **Details**: Sends a prompt to Llama for grammar correction while maintaining the meaning of the original transcription.  
   **Application**: Improves the quality of the transcribed text before it's used for queries.

### **`add_embeddings_to_chromadb()` Function**
   **Purpose**: Converts the fine-tuned transcription text into embeddings and stores them in ChromaDB.  
   **Details**: Uses `SentenceTransformer` to encode sentences and then stores them in a ChromaDB collection.  
   **Application**: Ensures that fine-tuned transcriptions can be queried based on their semantic content.

### **`get_relevant_context_from_chromadb()` Function**
   **Purpose**: Retrieves relevant context from ChromaDB based on a user query.  
   **Details**: Queries the embeddings stored in ChromaDB and uses cosine similarity to find the most relevant sentences.  
   **Application**: Provides contextual information to Llama for answering user queries based on fine-tuned transcription.

---

## **4. Models and Libraries Used**

### **SentenceTransformer**
   **Description**: A transformer-based model used to convert sentences into vector representations (embeddings). It utilizes models like **DistilRoBERTa** and **BERT** for encoding.  
   **Reference**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*.

### **Llama (via Ollama)**
   **Description**: Llama is a powerful language model used for text fine-tuning and answering queries. It is employed here to improve the grammar and clarity of transcribed text.  
   **Reference**: Llama models are continuously evolving, with improvements coming from OpenAI and other major research groups.

### **Whisper**
   **Description**: OpenAI’s Whisper model is used to transcribe speech from videos into text. It is robust across a variety of languages and noise levels.  
   **Reference**: Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. In *Proceedings of NeurIPS 2021*.

### **ChromaDB**
   **Description**: ChromaDB is a vector database optimized for handling document embeddings. It provides fast retrieval of embeddings for tasks like semantic search.  
   **Reference**: Chroma is an open-source vector database optimized for AI applications.

---

## **5. Suggested Features & Applications**

### **1. Multilingual Support**
   Add support for transcribing and fine-tuning content in multiple languages, increasing the app’s accessibility and user base globally.

### **2. Advanced Search Filters**
   Integrate advanced filters in the Q&A section for narrowing down queries by topics, relevance, or metadata like timestamps.

### **3. Real-Time Transcription**
   Allow for the transcription of live videos or audio streams in real-time, benefiting applications in education, journalism, and conferencing.

### **4. Voice Query Interface**
   Implement voice interaction capabilities, allowing users to ask questions via speech instead of typing. Integrate speech-to-text functionality for this feature.

### **5. Content Summarization**
   Introduce a summarization tool for users to get concise versions of long transcriptions, helping them get a quick gist of the content.

### **6. Video Library**
   Allow users to upload multiple videos, store them in a library, and provide the ability to search through transcriptions and associated embeddings.

### **7. Integration with Learning Management Systems (LMS)**
   Integrate the app with LMS platforms like Moodle, Canvas, or Blackboard, enabling seamless access to transcriptions and interactive Q&A for educational purposes.

---

## **6. Usage Example**

### **1. Upload Video**
   Users upload a video (in `.mp4`, `.mov`, or `.avi` formats).

### **2. Transcribe the Video**
   The app uses the **Whisper model** to transcribe the audio into text.

### **3. Fine-tune the Transcription**
   The transcribed text is fine-tuned using the **Llama model** to improve grammar and clarity.

### **4. Store Embeddings**
   The fine-tuned text is split into sentences, and **SentenceTransformer** is used to generate sentence embeddings. These embeddings are then stored in **ChromaDB** for later querying.

### **5. Ask Questions**
   The user can then ask questions about the transcription. The app retrieves relevant context from **ChromaDB** and uses **Llama** to generate an answer based on the transcriptions.

### **6. Evaluation metrics**

Long-form question answering (LFQA) enables answering a wide range of questions, but its flexibility poses enormous challenges for evaluation. We perform the first targeted study of the evaluation of long-form answers, covering both human and automatic evaluation practices. We hire domain experts in seven areas to provide preference judgments over pairs of answers, along with free-form justifications for their choices. We present a careful analysis of experts' evaluation, which focuses on new aspects such as the comprehensiveness of the answer. Next, we examine automatic text generation metrics, finding that no existing metrics are predictive of human preference judgments. However, some metrics correlate with fine-grained aspects of answers (e.g., coherence). We encourage future work to move away from a single "overall score" of the answer and adopt a multi-faceted evaluation, targeting aspects such as factuality and completeness. We publicly release all of our annotations and code to spur future work into LFQA evaluation.

https://arxiv.org/abs/2305.18201

---

## **7. About This App**

This app uses the **[vid2cleantxt](https://github.com/pszemraj/vid2cleantxt)** package for transcription and **Ollama's Llama model** for fine-tuning and interactive Q&A with **ChromaDB**. Developed by **[Prasoon Majumdar](https://github.com/thecalmmatter)**.

---

### **Final Notes**
This app provides an interactive solution for transcribing videos, fine-tuning the resulting text, and using it in a question-answering format. It has several applications, especially in educational and professional settings where video content needs to be analyzed and queried efficiently.
