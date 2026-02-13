# Vehicle Specification Extraction System with RAG

## Overview

The goal of this project is to build a domain-specific intelligent system that extracts structured vehicle specifications (e.g., torque values, fluid capacities, part numbers) from Automotive Service Manual PDFs using natural language queries.Unlike general-purpose PDF chatbots, this system is optimized specifically for automotive specification extraction. It leverages a Hybrid Retrieval-Augmented Generation (RAG) architecture to ensure accurate, structured, and deterministic outputs in JSON format.The application provides a clean Streamlit-based interface where users can upload service manual PDFs and query them using natural language.




https://github.com/user-attachments/assets/943e88cf-d3a3-4f66-98ce-a6febf994dac




## System Architecture Design

### Workflow

1. **PDF Upload**: Automotive service manual PDFs are uploaded via the Streamlit UI.
2. **Section-Based Chunking**: Documents are split into logical SECTION blocks
3. **Hybrid Retrieval**: Step 1: Keyword filtering (SPECIFICATIONS, case-insensitive). Step 2: Embedding similarity ranking using FAISS. Top-k relevant sections are selected.
4. **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (Hugging Face)
5. **Vector Store**: FAISS (Facebook AI Similarity Search)
6. **LLM Extraction**: OpenAI GPT models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo). Temperature set to 0 for deterministic extraction. Strict JSON output enforced via prompt engineering
7. **Structured Output**: Output returned as structured JSON:
```
[
  {
    "Section": "SECTION 204-01A: Front Suspension — Rear Wheel Drive (RWD)",
    "component": "Brake disc shield bolts",
    "spec_type": "Torque",
    "value": "17",
    "unit": "Nm"
  }
]
```

## Getting Started

To use the Vehicle Specification Extraction System:

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/parthvyas-ai/Automotive-vehicle-spec-extractor.git
   ```

2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application.
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501` to access the user interface.


## Acknowledgments

I would like to express my gratitude to the Hugging Face community for the all-MiniLM-L6-v2 Embeddings model, and OpenAI for providing the GPT-3.5 Turbo model through their API.

---

Feel free to explore and enhance the capabilities of the Vehicle Specification Extraction System. Happy querying!
